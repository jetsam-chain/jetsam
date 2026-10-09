// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The node's v1.5 client objects (M3.8): registered matrices, received
//! client proofs, and held registrations.
//!
//! - **Registered matrices.** Every node needs the matrix of every carried
//!   registry entry to judge a v1.5 tip (its live lanes are evaluated on
//!   them). They live on disk under `client-objects/matrices/`, one canonical
//!   `FieldR1cs` artifact per entry, and in RAM in the shared
//!   [`HistoryStepClientMatrixSet`] the verifier reads. A missing one is
//!   fetched from peers: the manifest (chunk digests, checked against the
//!   registered file root), then 1 MiB chunks (each checked against its
//!   digest by the transport, as it arrives), from any mix of peers;
//!   complete, the file must decode as a matrix of the client form whose
//!   structural digest is the registered `D`. A peer that serves bytes
//!   failing their identity is never asked again for that file.
//! - **Authenticated once, off the event loop.** The structural digest of a
//!   256 MiB matrix is minutes of CPU on two cores, and every node receives a
//!   registration at the same time: a completed file is handed back
//!   ([`ClientObjectFetched::MatrixAssembled`]) and authenticated on a
//!   blocking worker ([`ClientObjects::authenticate_assembled`]), its chunks
//!   hashed once (by the transport) and its matrix digested once. After a
//!   restart the files held on disk are listed without being read and
//!   authenticated again the same way ([`ClientObjects::authenticate_held_files`]).
//!   Until then a matrix is neither held for the verifier nor served (a tip
//!   that needs it has no verdict yet), nor fetched again.
//! - **Never in the way of blocks.** A matrix is decoded and digested on its
//!   own workers, at the lowest CPU priority, whoever asks: never on the
//!   shared pool where blocks are verified and produced
//!   (`on_authentication_workers`).
//! - **Client proofs.** Announced on gossip, fetched as bundles (submission,
//!   paying transaction, proof), decoded under their bounds, pre-passed once
//!   (`PreparedHistoryStepClient`: native verification, lincheck on `D`,
//!   fold) and queued (`ClientSubmissionQueue`): the miner's choice. The
//!   bundle bytes are kept on disk under `client-objects/proofs/`, so a node
//!   that restarts keeps its queue, and served to peers.
//! - **Registrations** submitted to this node (RPC) are held with their
//!   paying transaction until a block carries them, and relayed whole to
//!   peers, which hold them for their own miners (M3.10, §8 gap b).
//!
//! This node serves what it holds ([`ClientObjectSource`]).

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use jetsam_chain::consensus::client_objects::{
    matrix_file_chunk_digest, matrix_file_root_from_chunk_digests, queue::ClientCandidate,
    queue::ClientSubmissionQueue, ClientObject, ClientObjectRules, ClientRegistration,
    ClientRegistryState, ClientSubmission, CLIENT_MATRIX_CHUNK_BYTES,
    CLIENT_MATRIX_MAX_FILE_BYTES,
};
use jetsam_ivc_core::field_r1cs::FieldR1cs;
use jetsam_p2p::client_object_codec::ClientObjectRequest;
use jetsam_p2p::client_object_protocol::{
    ClientProofAnnouncement, ClientProofBundle, ClientProofFetcher, ClientProofId, Hash32,
    MatrixFileId,
};
use jetsam_p2p::client_object_transport::ClientObjectSource;
use jetsam_recursive::acceptance::history_step::{
    HistoryStepClientRegistry, HistoryStepClientWitness, PreparedHistoryStepClient,
};
use jetsam_recursive::{
    AuthenticatedClientMatrix, HistoryStepClientForm, HistoryStepClientMatrixSet,
};
use jetsam_tx::PagedSpendIntent;
use libp2p::PeerId;

/// Client proofs queued for the miner.
const CLIENT_QUEUE_CAPACITY: usize = 64;
/// Proofs wanted at once, providers per proof, requests per peer.
const PROOF_FETCH_WANTED: usize = 64;
const PROOF_FETCH_PROVIDERS: usize = 8;
const PROOF_FETCH_PER_PEER: usize = 2;
/// Chunk requests in flight per matrix file.
const MATRIX_CHUNKS_IN_FLIGHT: usize = 4;
/// A request without an answer is retried from another peer after this.
const CLIENT_OBJECT_REQUEST_DEADLINE: Duration = Duration::from_secs(90);
/// Held registrations (one per block is mined; a registry holds 16).
const MAX_HELD_REGISTRATIONS: usize = 16;
/// Matrix files fetched at once: a registry holds 16 (review B3: a forged
/// snapshot manifest wanted as many as it listed, without bound).
const MAX_MATRIX_FETCHES: usize =
    jetsam_chain::consensus::client_objects::CLIENT_REGISTRY_CAPACITY;
/// Where a matrix file found on disk that no rule allows this node to keep
/// is moved at startup, never deleted (`client-objects/matrices-set-aside`).
const SET_ASIDE_DIR: &str = "matrices-set-aside";

/// What went wrong with a client object handed to this node.
#[derive(Debug, thiserror::Error)]
pub enum ClientObjectsError {
    #[error("client objects are not active on this node (no v1.5 client form)")]
    Inactive,
    #[error("matrix file: {0}")]
    MatrixFile(&'static str),
    #[error("client proof bundle: {0}")]
    Bundle(String),
    #[error("client {0} is not registered, or its matrix is not held yet")]
    UnknownClient(String),
    #[error(
        "client {matrix_digest} is registered but not active yet: the chain is at height \
         {tip_height}, the client is active from height {active_from} (no block may carry \
         its proof before); submit it again once the chain reaches height {}",
        active_from.saturating_sub(1)
    )]
    NotYetActive {
        matrix_digest: String,
        tip_height: u64,
        active_from: u64,
    },
    #[error("client proof does not verify: {0}")]
    Proof(String),
    #[error("registration: {0}")]
    Registration(String),
    #[error("queue: {0:?}")]
    Queue(jetsam_chain::consensus::client_objects::queue::ClientQueueError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// One registration held for this node's miner: the object and the
/// transaction that pays its license (and carries its marker).
#[derive(Clone, Debug)]
pub struct HeldRegistration {
    pub registration: ClientRegistration,
    pub payment: PagedSpendIntent,
}

/// A client object request this node sends, with its node-local token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientObjectFetch {
    pub token: u64,
    pub peer: PeerId,
    pub request: ClientObjectRequest,
}

/// What a fetched client object amounted to.
#[derive(Debug)]
pub enum ClientObjectFetched {
    /// A client proof bundle, exactly the announced bytes: to receive
    /// ([`ClientObjects::receive_bundle`], CPU-heavy).
    Bundle(Vec<u8>),
    /// Every chunk of a registered matrix file arrived: to authenticate off
    /// the event loop ([`ClientObjects::authenticate_assembled`], CPU-heavy).
    MatrixAssembled(AssembledMatrixFile),
    /// Progress (a manifest or chunk stored), or a late or unwanted answer.
    Nothing,
    /// The peer served bytes failing their identity: penalise it.
    Liar,
}

/// A registered matrix file whose every chunk arrived, each the one its
/// manifest lists, the manifest the registered root's; not authenticated yet
/// (made only by [`ClientObjects::on_fetched`]).
pub struct AssembledMatrixFile {
    id: MatrixFileId,
    digests: Vec<Hash32>,
    chunks: Vec<Vec<u8>>,
}

impl AssembledMatrixFile {
    pub fn id(&self) -> MatrixFileId {
        self.id
    }
}

impl std::fmt::Debug for AssembledMatrixFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AssembledMatrixFile")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

#[derive(Default)]
struct MatrixFetch {
    manifest: Option<Vec<Hash32>>,
    chunks: Vec<Option<Vec<u8>>>,
    /// In flight: chunk index (or `u32::MAX` for the manifest) → peer, since.
    in_flight: BTreeMap<u32, (PeerId, Instant)>,
    /// Peers that served false bytes for this file.
    excluded: BTreeSet<PeerId>,
    cursor: usize,
}

struct State {
    held_files: BTreeMap<MatrixFileId, PathBuf>,
    manifests: BTreeMap<MatrixFileId, Vec<u8>>,
    fetches: BTreeMap<MatrixFileId, MatrixFetch>,
    /// Files on disk found by `open`, not authenticated again yet.
    unauthenticated_files: BTreeMap<MatrixFileId, PathBuf>,
    /// Files being authenticated off the event loop (assembled, or read
    /// back after a restart).
    authenticating: BTreeSet<MatrixFileId>,
    next_token: u64,
    tokens: BTreeMap<u64, (PeerId, ClientObjectRequest, Instant)>,
    proofs: ClientProofFetcher<PeerId>,
    bundles: BTreeMap<ClientProofId, Arc<[u8]>>,
    queue: ClientSubmissionQueue<Arc<PreparedHistoryStepClient>>,
    registrations: BTreeMap<Hash32, HeldRegistration>,
}

/// The node's client objects (see the module documentation).
pub struct ClientObjects {
    form: HistoryStepClientForm,
    matrices: Arc<HistoryStepClientMatrixSet>,
    matrix_dir: PathBuf,
    proof_dir: PathBuf,
    /// Held registrations, one file each (the relayed notice bytes), so a
    /// restart keeps what this node's miner was asked to include (M3.10).
    registration_dir: PathBuf,
    state: Mutex<State>,
    /// A verification waits for a matrix still to be fetched or
    /// authenticated ([`Self::note_matrices_needed`]); cleared once none is.
    needed: AtomicBool,
    /// The rules a matrix file must satisfy to be fetched or kept: its `D`
    /// in the catalogue, its length within the cap (review B3).
    rules: ClientObjectRules,
}

fn registration_file_name(matrix_digest: &Hash32) -> String {
    format!("{}.registration", hex32(matrix_digest))
}

fn hex32(bytes: &[u8; 32]) -> String {
    hex::encode(bytes)
}

fn matrix_file_name(id: &MatrixFileId) -> String {
    format!(
        "{}.{}.{}.matrix",
        hex32(&id.matrix_digest),
        hex32(&id.file_root),
        id.file_len
    )
}

fn parse_matrix_file_name(name: &str) -> Option<MatrixFileId> {
    let mut parts = name.strip_suffix(".matrix")?.split('.');
    let digest: [u8; 32] = hex::decode(parts.next()?).ok()?.try_into().ok()?;
    let root: [u8; 32] = hex::decode(parts.next()?).ok()?.try_into().ok()?;
    let len: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(MatrixFileId {
        matrix_digest: digest,
        file_root: root,
        file_len: len,
    })
}

/// Which workers authenticate a matrix (see [`on_authentication_workers`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AuthenticationPriority {
    /// Nothing waits for it: the lowest CPU priority, never in the way of a
    /// block's verification.
    Background,
    /// A verification answered "client matrix unavailable" while it was
    /// pending: the node makes no progress until it is held, so it runs at
    /// the node's own priority.
    Needed,
}

/// Run `operation` (a matrix's decode, structural digest or chunk hashes:
/// minutes of CPU at 256 MiB on two cores, on every node at once when a
/// registration is mined) on the workers that authenticate matrices at
/// `priority`, and its parallel parts with it.
///
/// Never on the shared pool where blocks are verified and produced: Rayon
/// runs a job handed to a pool from outside only when its workers find
/// nothing else to do, so a block's verification queued behind an
/// authentication ended with it (testnet 3, 2026-10-09: block 1222 verified
/// in 102.6 s instead of 9-11 s on the two-vCPU seed, during the 95 s
/// authentication of a 203 MiB matrix). As many workers as the shared pool,
/// so an idle node authenticates as fast as before.
///
/// [`AuthenticationPriority::Background`] workers run at the lowest CPU
/// priority Linux has (SCHED_IDLE, below nice 19, which they take as well):
/// a CPU runs them when nothing else wants it, so a core that verifies or
/// produces a block leaves them only what it does not use. That is right
/// for a matrix nothing waits for, and wrong for one the node's progress
/// depends on: a two-vCPU node restarted behind the network judged its
/// snapshot candidate again and again, refused each time for want of the
/// matrix, and those judgements left the authentication about 15 % of the
/// CPUs (testnet 3, 2026-10-09: 630 s instead of 125 s). A thread cannot
/// leave SCHED_IDLE without privilege, so [`AuthenticationPriority::Needed`]
/// work runs on workers of its own that keep the node's priority (both sets
/// are started together by the first caller, never by one of these workers,
/// whose priority a thread it starts would inherit). On other systems both
/// keep the default priority. If workers cannot be started, the caller's
/// thread does the work.
fn on_authentication_workers<R: Send>(
    priority: AuthenticationPriority,
    operation: impl FnOnce() -> R + Send,
) -> R {
    static WORKERS: OnceLock<[Option<rayon::ThreadPool>; 2]> = OnceLock::new();
    let [background, needed] = WORKERS.get_or_init(|| {
        let threads = jetsam_miner::configured_process_cpu_budget()
            .map(|plan| plan.shared_pool_threads)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
        let start = |name: &'static str, lowest: bool| {
            let builder = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(move |index| format!("{name}-{index}"));
            let builder = if lowest {
                builder.start_handler(|_| lowest_cpu_priority())
            } else {
                builder
            };
            builder
                .build()
                .map_err(|error| {
                    tracing::warn!(
                        %error,
                        workers = name,
                        "client matrix authentication workers not started: matrices are \
                         authenticated on the caller's thread"
                    )
                })
                .ok()
        };
        [
            start("jetsam-matrix-auth", true),
            start("jetsam-matrix-need", false),
        ]
    });
    let workers = match priority {
        AuthenticationPriority::Background => background,
        AuthenticationPriority::Needed => needed,
    };
    match workers {
        Some(workers) => workers.install(operation),
        None => operation(),
    }
}

/// The calling thread, and it alone, to the lowest CPU priority: nice 19,
/// then SCHED_IDLE. Nice 19 alone left these workers more than its weight
/// (measured: a quarter to a third of two CPUs while Rayon workers at
/// nice 10 verified on them); SCHED_IDLE leaves them what the CPUs do not
/// use (under 8 % there).
#[cfg(target_os = "linux")]
fn lowest_cpu_priority() {
    // `who = 0` and `pid = 0` are the calling thread: Linux keeps the nice
    // value and the policy per thread. Lowering one's own priority needs no
    // privilege.
    // SAFETY: setpriority reads no memory of ours.
    if unsafe { libc::setpriority(libc::PRIO_PROCESS as _, 0, 19) } != 0 {
        tracing::warn!(
            error = %std::io::Error::last_os_error(),
            "a client matrix authentication worker keeps the default CPU priority"
        );
    }
    let idle = libc::sched_param { sched_priority: 0 };
    // SAFETY: `idle` outlives the call, which only reads it.
    if unsafe { libc::sched_setscheduler(0, libc::SCHED_IDLE, &idle) } != 0 {
        tracing::warn!(
            error = %std::io::Error::last_os_error(),
            "a client matrix authentication worker keeps the default CPU policy"
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn lowest_cpu_priority() {}

/// The digests of a matrix file's chunks, in order (its manifest; the root is
/// their hash): one pass over the file, its chunks hashed in parallel on the
/// authentication workers.
fn matrix_chunk_digests(priority: AuthenticationPriority, file: &[u8]) -> Vec<Hash32> {
    use rayon::prelude::*;
    on_authentication_workers(priority, || {
        file.par_chunks(CLIENT_MATRIX_CHUNK_BYTES)
            .enumerate()
            .map(|(index, chunk)| matrix_file_chunk_digest(index as u32, chunk))
            .collect()
    })
}

/// Where a file is written before it takes the name `path`.
fn staged_path(path: &Path) -> PathBuf {
    path.with_extension(format!("tmp-{}", std::process::id()))
}

/// Write `parts` to `staged`, in order, synced to disk; each part is freed
/// once written.
fn write_staged<P: AsRef<[u8]>>(
    staged: &Path,
    parts: impl IntoIterator<Item = P>,
) -> std::io::Result<()> {
    let mut file = std::fs::File::create(staged)?;
    for part in parts {
        file.write_all(part.as_ref())?;
    }
    file.sync_all()
}

fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let staged = staged_path(path);
    write_staged(&staged, [bytes])?;
    std::fs::rename(&staged, path)
}

impl ClientObjects {
    /// Open the client objects under `data_dir`: every registration held,
    /// re-checked under `rules`, and the list of the matrix files held, not
    /// read here: each is authenticated again (decoded as a matrix of `form`
    /// whose structural digest is the one its name registers) into `matrices`
    /// by [`Self::authenticate_held_files`], off the event loop.
    pub fn open(
        data_dir: &Path,
        form: &HistoryStepClientForm,
        matrices: Arc<HistoryStepClientMatrixSet>,
        rules: &ClientObjectRules,
    ) -> Result<Self, ClientObjectsError> {
        let root = data_dir.join("client-objects");
        let matrix_dir = root.join("matrices");
        let proof_dir = root.join("proofs");
        let registration_dir = root.join("registrations");
        std::fs::create_dir_all(&matrix_dir)?;
        std::fs::create_dir_all(&proof_dir)?;
        std::fs::create_dir_all(&registration_dir)?;
        let objects = Self {
            form: form.clone(),
            matrices,
            matrix_dir,
            proof_dir,
            registration_dir,
            state: Mutex::new(State {
                held_files: BTreeMap::new(),
                manifests: BTreeMap::new(),
                fetches: BTreeMap::new(),
                unauthenticated_files: BTreeMap::new(),
                authenticating: BTreeSet::new(),
                next_token: 1,
                tokens: BTreeMap::new(),
                proofs: ClientProofFetcher::new(
                    PROOF_FETCH_WANTED,
                    PROOF_FETCH_PROVIDERS,
                    PROOF_FETCH_PER_PEER,
                ),
                bundles: BTreeMap::new(),
                queue: ClientSubmissionQueue::new(CLIENT_QUEUE_CAPACITY),
                registrations: BTreeMap::new(),
            }),
            needed: AtomicBool::new(false),
            rules: *rules,
        };
        for entry in std::fs::read_dir(&objects.matrix_dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(id) = parse_matrix_file_name(&name) else {
                continue;
            };
            // A file no rule lets this node keep (its `D` outside the
            // catalogue, a length beyond the cap) is never authenticated
            // again: moved aside, not deleted (review B3).
            if !objects.matrix_file_admissible(&id) {
                objects.set_aside(&id, &entry.path(), "its D is outside the client catalogue");
                continue;
            }
            objects
                .state
                .lock()
                .expect("client objects lock")
                .unauthenticated_files
                .insert(id, entry.path());
        }
        // Held registrations kept before a restart: each re-checked as a
        // relayed notice is (its payment opens and pays it, its `D` is in
        // the catalogue).
        for entry in std::fs::read_dir(&objects.registration_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|ext| ext.to_str()) != Some("registration") {
                continue;
            }
            let bytes = read_bounded(
                &entry.path(),
                jetsam_p2p::client_object_protocol::MAX_CLIENT_REGISTRATION_NOTICE_BYTES as u64,
            )?;
            match jetsam_p2p::client_object_protocol::ClientRegistrationNotice::decode(&bytes, rules)
            {
                Ok(notice) => {
                    objects
                        .state
                        .lock()
                        .expect("client objects lock")
                        .registrations
                        .insert(
                            notice.registration.matrix_digest,
                            HeldRegistration {
                                registration: notice.registration,
                                payment: notice.payment,
                            },
                        );
                }
                Err(error) => {
                    tracing::warn!(file = %entry.path().display(), %error, "kept client registration dropped");
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        Ok(objects)
    }

    pub fn form(&self) -> &HistoryStepClientForm {
        &self.form
    }

    /// Whether the rules let this node fetch or keep the matrix file `id`:
    /// a non-null `D` listed in the client catalogue and a non-null root, a
    /// length within the cap of a registration (review B3: a forged snapshot
    /// manifest had any `D`, of any length, fetched, kept and authenticated
    /// again at every start).
    fn matrix_file_admissible(&self, id: &MatrixFileId) -> bool {
        id.matrix_digest != [0u8; 32]
            && id.file_root != [0u8; 32]
            && id.file_len != 0
            && id.file_len <= self.rules.max_matrix_file_bytes.min(CLIENT_MATRIX_MAX_FILE_BYTES)
            && self.rules.catalogue.contains(&id.matrix_digest)
    }

    /// Move the matrix file `id` at `path` aside (never delete it): it is
    /// neither authenticated nor held nor served. Logged at WARN with why.
    fn set_aside(&self, id: &MatrixFileId, path: &Path, why: &str) {
        let aside = self
            .matrix_dir
            .parent()
            .map_or_else(|| self.matrix_dir.join(SET_ASIDE_DIR), |root| root.join(SET_ASIDE_DIR));
        let moved = std::fs::create_dir_all(&aside).and_then(|()| {
            let target = aside.join(path.file_name().unwrap_or_default());
            std::fs::rename(path, &target).map(|()| target)
        });
        match moved {
            Ok(target) => tracing::warn!(
                matrix_digest = %hex32(&id.matrix_digest),
                bytes = id.file_len,
                file = %target.display(),
                why,
                "client matrix file set aside: this node does not keep it"
            ),
            Err(error) => tracing::error!(
                matrix_digest = %hex32(&id.matrix_digest),
                file = %path.display(),
                %error,
                why,
                "client matrix file could not be set aside: it is ignored, not deleted"
            ),
        }
    }

    /// At startup, before any matrix file found on disk is authenticated
    /// again: every file that is neither an entry of the canonical chain's
    /// `registry` nor the file of a registration this node holds for its
    /// miner is set aside, not deleted (review B3). Returns how many.
    pub fn set_aside_unregistered_files(&self, registry: &ClientRegistryState) -> usize {
        let unregistered: Vec<(MatrixFileId, PathBuf)> = {
            let mut state = self.state.lock().expect("client objects lock");
            let held: BTreeSet<MatrixFileId> = state
                .registrations
                .values()
                .map(|held| MatrixFileId::of_registration(&held.registration))
                .collect();
            let canonical: BTreeSet<MatrixFileId> =
                registry.entries().iter().map(MatrixFileId::of_entry).collect();
            let unregistered: Vec<MatrixFileId> = state
                .unauthenticated_files
                .keys()
                .filter(|id| !canonical.contains(id) && !held.contains(id))
                .copied()
                .collect();
            let unregistered = unregistered
                .into_iter()
                .filter_map(|id| state.unauthenticated_files.remove(&id).map(|path| (id, path)))
                .collect();
            self.settle_needed(&state);
            unregistered
        };
        for (id, path) in &unregistered {
            self.set_aside(id, path, "no entry of the canonical chain's registry names it");
        }
        unregistered.len()
    }

    pub fn matrices(&self) -> &Arc<HistoryStepClientMatrixSet> {
        &self.matrices
    }

    /// The workers the next stage of an authentication runs on: asked again
    /// at each stage, so a matrix a verification starts waiting for moves to
    /// the node's priority at its next stage.
    fn authentication_priority(&self) -> AuthenticationPriority {
        if self.needed.load(Ordering::Acquire) {
            AuthenticationPriority::Needed
        } else {
            AuthenticationPriority::Background
        }
    }

    /// A verification answered "client matrix unavailable" (M3.8: no
    /// verdict): the node waits for a matrix it has yet to fetch or
    /// authenticate, so every matrix still to be authenticated is from now on
    /// authenticated at the node's own CPU priority, not in the background,
    /// until none is left. Returns whether one is (nothing to hurry
    /// otherwise). Logged at INFO when it starts.
    pub fn note_matrices_needed(&self) -> bool {
        let state = self.state.lock().expect("client objects lock");
        let pending = Self::pending_matrices(&state);
        if pending == 0 {
            return false;
        }
        if !self.needed.swap(true, Ordering::AcqRel) {
            tracing::info!(
                pending,
                "a verification waits for client matrices not held yet: they are authenticated \
                 at the node's CPU priority until they are"
            );
        }
        true
    }

    /// Whether a matrix is still to be fetched or authenticated (a snapshot
    /// candidate refused for want of one waits for this to end).
    pub fn awaits_matrices(&self) -> bool {
        Self::pending_matrices(&self.state.lock().expect("client objects lock")) > 0
    }

    /// Matrix files wanted and not held yet: being fetched, found on disk
    /// and not authenticated again yet, or being authenticated.
    fn pending_matrices(state: &State) -> usize {
        state.fetches.len() + state.unauthenticated_files.len() + state.authenticating.len()
    }

    /// Back to the background once no matrix is pending.
    fn settle_needed(&self, state: &State) {
        if Self::pending_matrices(state) == 0 {
            self.needed.store(false, Ordering::Release);
        }
    }

    /// The matrix file read from `artifact` decoded as a matrix of the client
    /// form, its `D` computed (the one structural pass), on the
    /// authentication workers.
    fn open_client_matrix(
        &self,
        artifact: &mut (impl Read + Send),
    ) -> Option<AuthenticatedClientMatrix> {
        let matrix = on_authentication_workers(self.authentication_priority(), || {
            FieldR1cs::read_artifact_unbound(
                artifact,
                self.form.shape(),
                CLIENT_MATRIX_MAX_FILE_BYTES as usize,
            )
            .ok()
        })?;
        on_authentication_workers(self.authentication_priority(), || {
            self.matrices.authenticate(Arc::new(matrix)).ok()
        })
    }

    /// The matrix file read from `artifact` decoded as the matrix of the
    /// registered `D` of `id`.
    fn open_registered_matrix(
        &self,
        id: &MatrixFileId,
        artifact: &mut (impl Read + Send),
    ) -> Result<AuthenticatedClientMatrix, ClientObjectsError> {
        self.open_client_matrix(artifact)
            .filter(|matrix| matrix.digest() == id.matrix_digest)
            .ok_or(ClientObjectsError::MatrixFile(
                "not a matrix of the client form with this D",
            ))
    }

    /// The file `bytes` as the registered file `id`: its length and root,
    /// then a matrix of the client form whose structural digest is `D`. One
    /// pass of each hash; returns the matrix and the chunk digests.
    fn authenticate_matrix_file(
        &self,
        id: &MatrixFileId,
        bytes: &[u8],
    ) -> Result<(AuthenticatedClientMatrix, Vec<Hash32>), ClientObjectsError> {
        if bytes.len() != id.file_len as usize || bytes.is_empty() {
            return Err(ClientObjectsError::MatrixFile("length is not the registered one"));
        }
        let digests = matrix_chunk_digests(self.authentication_priority(), bytes);
        if matrix_file_root_from_chunk_digests(u64::from(id.file_len), &digests) != id.file_root {
            return Err(ClientObjectsError::MatrixFile("root is not the registered one"));
        }
        Ok((self.open_registered_matrix(id, &mut &bytes[..])?, digests))
    }

    /// The file at `path` as the registered file `id`, read from disk: its
    /// length, its root (its chunks read and hashed a few at a time, in
    /// parallel), then a matrix of the client form whose structural digest
    /// is `D`, decoded from the file. Its bytes are never in memory whole,
    /// nor beside its decoded matrix (three times their size) for the whole
    /// structural digest. Returns the matrix and the chunk digests.
    fn authenticate_held_file(
        &self,
        id: &MatrixFileId,
        path: &Path,
    ) -> Result<(AuthenticatedClientMatrix, Vec<Hash32>), ClientObjectsError> {
        let mut file = std::fs::File::open(path)?;
        if file.metadata()?.len() != u64::from(id.file_len) || id.file_len == 0 {
            return Err(ClientObjectsError::MatrixFile("length is not the registered one"));
        }
        // A few chunks at a time, each batch on the workers of the moment.
        let mut digests = Vec::with_capacity(id.chunk_count());
        let mut chunks = Vec::new();
        while digests.len() < id.chunk_count() {
            on_authentication_workers(
                self.authentication_priority(),
                || -> std::io::Result<()> {
                    use rayon::prelude::*;
                    let first = digests.len();
                    chunks.clear();
                    for index in first..id.chunk_count().min(first + rayon::current_num_threads()) {
                        let mut chunk =
                            vec![0u8; id.chunk_len(index as u32).expect("a chunk of id")];
                        file.read_exact(&mut chunk)?;
                        chunks.push(chunk);
                    }
                    digests.par_extend(chunks.par_iter().enumerate().map(|(offset, chunk)| {
                        matrix_file_chunk_digest((first + offset) as u32, chunk)
                    }));
                    Ok(())
                },
            )?;
        }
        if matrix_file_root_from_chunk_digests(u64::from(id.file_len), &digests) != id.file_root {
            return Err(ClientObjectsError::MatrixFile("root is not the registered one"));
        }
        std::io::Seek::rewind(&mut file)?;
        let matrix = self.open_registered_matrix(id, &mut std::io::BufReader::new(file))?;
        Ok((matrix, digests))
    }

    /// Hold the authenticated matrix file `id` kept at `path`: inserted for
    /// the verifier, its manifest (`digests`) and chunks served to peers.
    fn hold_authenticated(
        &self,
        id: MatrixFileId,
        matrix: AuthenticatedClientMatrix,
        digests: Vec<Hash32>,
        path: PathBuf,
    ) -> Result<(), ClientObjectsError> {
        // Every path that keeps a matrix ends here: none keeps one the rules
        // do not allow (review B3).
        if !self.matrix_file_admissible(&id) {
            return Err(ClientObjectsError::MatrixFile(
                "its D is outside the client catalogue, or its length beyond the cap",
            ));
        }
        self.matrices
            .insert_authenticated(matrix)
            .map_err(|_| ClientObjectsError::MatrixFile("form shape"))?;
        let mut state = self.state.lock().expect("client objects lock");
        state.manifests.insert(id, digests.concat());
        state.held_files.insert(id, path);
        state.fetches.remove(&id);
        state.authenticating.remove(&id);
        // Held anew (`registerClient`) before the restart's pass reached it.
        state.unauthenticated_files.remove(&id);
        self.settle_needed(&state);
        Ok(())
    }

    /// Hold the registered matrix file `id` (`bytes`, authenticated here):
    /// written to disk, inserted for the verifier, served to peers.
    pub fn insert_matrix_file(
        &self,
        id: MatrixFileId,
        bytes: &[u8],
    ) -> Result<(), ClientObjectsError> {
        let (matrix, digests) = self.authenticate_matrix_file(&id, bytes)?;
        let path = self.matrix_dir.join(matrix_file_name(&id));
        write_atomically(&path, bytes)?;
        self.hold_authenticated(id, matrix, digests, path)
    }

    /// Authenticate a matrix file whose chunks were fetched (CPU-heavy: call
    /// it off the event loop) and hold it. Every chunk is the one the
    /// registered root lists (checked as it arrived), so only the structural
    /// identity is left: refused if the file is not a matrix of the
    /// registered `D` (no peer can do better: the fetch is dropped).
    pub fn authenticate_assembled(
        &self,
        assembled: AssembledMatrixFile,
    ) -> Result<MatrixFileId, ClientObjectsError> {
        let id = assembled.id;
        match self.hold_assembled(assembled) {
            Ok(()) => Ok(id),
            Err(error) => {
                self.forget_assembled(&id);
                Err(error)
            }
        }
    }

    fn hold_assembled(&self, assembled: AssembledMatrixFile) -> Result<(), ClientObjectsError> {
        let AssembledMatrixFile {
            id,
            digests,
            chunks,
        } = assembled;
        if chunks.iter().map(Vec::len).sum::<usize>() != id.file_len as usize {
            return Err(ClientObjectsError::MatrixFile("length is not the registered one"));
        }
        // On disk before it is decoded, each chunk freed once written, then
        // decoded from there: the file is never copied whole in memory nor
        // held beside its decoded matrix, three times its size (0.62 GiB for
        // the 203 MiB file of a batch of 128) for the whole structural
        // digest. It takes its name once authenticated.
        let path = self.matrix_dir.join(matrix_file_name(&id));
        let staged = staged_path(&path);
        let authenticated = write_staged(&staged, chunks)
            .map_err(ClientObjectsError::from)
            .and_then(|()| {
                let mut artifact = std::io::BufReader::new(std::fs::File::open(&staged)?);
                self.open_registered_matrix(&id, &mut artifact)
            })
            .and_then(|matrix| {
                std::fs::rename(&staged, &path)?;
                Ok(matrix)
            });
        match authenticated {
            Ok(matrix) => self.hold_authenticated(id, matrix, digests, path),
            Err(error) => {
                let _ = std::fs::remove_file(&staged);
                Err(error)
            }
        }
    }

    /// An assembled file that will not be authenticated (its worker was
    /// refused or lost): no longer awaited, it is fetched again when wanted.
    pub fn forget_assembled(&self, id: &MatrixFileId) {
        let mut state = self.state.lock().expect("client objects lock");
        state.authenticating.remove(id);
        self.settle_needed(&state);
    }

    /// Authenticate again every matrix file `open` found on disk (CPU-heavy:
    /// call it off the event loop), one file at a time, and hold each. A file
    /// that no longer reads as its registered identity is dropped, to be
    /// fetched again. Returns the files held.
    pub fn authenticate_held_files(&self) -> usize {
        let mut held = 0usize;
        loop {
            let (id, path) = {
                let mut state = self.state.lock().expect("client objects lock");
                let Some((id, path)) = state.unauthenticated_files.pop_first() else {
                    break;
                };
                state.authenticating.insert(id);
                (id, path)
            };
            let outcome = self
                .authenticate_held_file(&id, &path)
                .and_then(|(matrix, digests)| {
                    self.hold_authenticated(id, matrix, digests, path.clone())
                });
            match outcome {
                Ok(()) => held += 1,
                Err(error) => {
                    tracing::warn!(
                        file = %path.display(),
                        %error,
                        "held client matrix file refused; it will be fetched again"
                    );
                    let _ = std::fs::remove_file(&path);
                    self.forget_assembled(&id);
                }
            }
        }
        held
    }

    /// Whether the matrix of `D` is held.
    pub fn holds_matrix(&self, matrix_digest: &Hash32) -> bool {
        self.matrices.holds(matrix_digest)
    }

    /// Fetch every registered matrix in `wanted` this node does not hold
    /// (the chain's registry, or a snapshot candidate's), nor has on disk or
    /// in hand awaiting authentication — only those the rules allow it to
    /// keep (`D` in the catalogue, length within the cap), and never more
    /// than a registry holds at once (review B3). Callers list the chain's
    /// registry first: what does not fit waits for a later call.
    pub fn want_matrices(&self, wanted: impl IntoIterator<Item = MatrixFileId>) {
        let mut state = self.state.lock().expect("client objects lock");
        for id in wanted {
            if !self.matrix_file_admissible(&id) {
                tracing::debug!(
                    matrix_digest = %hex32(&id.matrix_digest),
                    bytes = id.file_len,
                    "client matrix not fetched: its D is outside the client catalogue, or its \
                     length beyond the cap"
                );
                continue;
            }
            if !state.held_files.contains_key(&id)
                && !state.unauthenticated_files.contains_key(&id)
                && !state.authenticating.contains(&id)
                && !self.matrices.holds(&id.matrix_digest)
            {
                let fetching = state.fetches.len();
                if let std::collections::btree_map::Entry::Vacant(fetch) = state.fetches.entry(id) {
                    if fetching >= MAX_MATRIX_FETCHES {
                        tracing::debug!(
                            matrix_digest = %hex32(&id.matrix_digest),
                            fetching,
                            "client matrix not fetched yet: as many fetches as a registry holds"
                        );
                        continue;
                    }
                    fetch.insert(MatrixFetch::default());
                    tracing::info!(
                        matrix_digest = %hex32(&id.matrix_digest),
                        bytes = id.file_len,
                        "fetching registered client matrix from peers"
                    );
                }
            }
        }
    }

    /// [`Self::want_matrices`] for exactly `wanted`: a fetch no longer
    /// wanted (a snapshot candidate retired, an entry a reorg removed) is
    /// dropped, its chunks freed, so nothing is fetched for ever. Its
    /// requests in flight go with it: an answer arriving later is nothing,
    /// never judged against a new fetch of the same file (review 2, R1: an
    /// honest provider's late chunk, judged against a fetch with no manifest
    /// yet, excluded it as a liar).
    pub fn want_only_matrices(&self, wanted: &[MatrixFileId]) {
        {
            let mut state = self.state.lock().expect("client objects lock");
            let before = state.fetches.len();
            let State { fetches, tokens, .. } = &mut *state;
            fetches.retain(|id, _| wanted.contains(id));
            tokens.retain(|_, (_, request, _)| match request {
                ClientObjectRequest::MatrixManifest(file)
                | ClientObjectRequest::MatrixChunk { file, .. } => fetches.contains_key(file),
                ClientObjectRequest::Proof(_) => true,
            });
            if state.fetches.len() != before {
                tracing::info!(
                    dropped = before - state.fetches.len(),
                    "client matrix fetches no longer wanted dropped"
                );
            }
            self.settle_needed(&state);
        }
        self.want_matrices(wanted.iter().copied());
    }

    /// Matrix files being fetched.
    pub fn fetching_matrices(&self) -> Vec<MatrixFileId> {
        self.state
            .lock()
            .expect("client objects lock")
            .fetches
            .keys()
            .copied()
            .collect()
    }

    /// Record a client-proof announcement from `peer`.
    pub fn announced(&self, peer: PeerId, announcement: ClientProofAnnouncement) {
        let mut state = self.state.lock().expect("client objects lock");
        if state.bundles.contains_key(&announcement.id) {
            return;
        }
        state.proofs.announced(peer, announcement);
    }

    /// `peer` disconnected.
    pub fn forget_peer(&self, peer: &PeerId) {
        let mut state = self.state.lock().expect("client objects lock");
        state.proofs.forget_peer(peer);
        for fetch in state.fetches.values_mut() {
            fetch.in_flight.retain(|_, (owner, _)| owner != peer);
        }
        state.tokens.retain(|_, (owner, _, _)| owner != peer);
    }

    /// The requests to send now, to the `connected` peers.
    pub fn next_requests(&self, connected: &[PeerId], now: Instant) -> Vec<ClientObjectFetch> {
        let mut state = self.state.lock().expect("client objects lock");
        let mut out = Vec::new();
        // Requests answered by nobody are freed.
        let expired: Vec<u64> = state
            .tokens
            .iter()
            .filter(|(_, (_, _, since))| {
                now.saturating_duration_since(*since) >= CLIENT_OBJECT_REQUEST_DEADLINE
            })
            .map(|(token, _)| *token)
            .collect();
        for token in expired {
            if let Some((peer, request, _)) = state.tokens.remove(&token) {
                release(&mut state, &peer, &request, false);
            }
        }
        // Client proofs: the fetcher's policy.
        for (peer, id) in state.proofs.next_requests() {
            let token = state.next_token;
            state.next_token += 1;
            let request = ClientObjectRequest::Proof(id);
            state.tokens.insert(token, (peer, request, now));
            out.push(ClientObjectFetch {
                token,
                peer,
                request,
            });
        }
        if connected.is_empty() {
            return out;
        }
        // Matrices: the manifest first, then chunks, rotating over peers.
        let ids: Vec<MatrixFileId> = state.fetches.keys().copied().collect();
        for id in ids {
            let mut issued = Vec::new();
            {
                let fetch = state.fetches.get_mut(&id).expect("listed above");
                let mut pick = |fetch: &mut MatrixFetch| -> Option<PeerId> {
                    for _ in 0..connected.len() {
                        let peer = connected[fetch.cursor % connected.len()];
                        fetch.cursor = fetch.cursor.wrapping_add(1);
                        if !fetch.excluded.contains(&peer) {
                            return Some(peer);
                        }
                    }
                    None
                };
                match fetch.manifest.clone() {
                    None => {
                        if !fetch.in_flight.contains_key(&u32::MAX) {
                            if let Some(peer) = pick(fetch) {
                                fetch.in_flight.insert(u32::MAX, (peer, now));
                                issued.push((peer, ClientObjectRequest::MatrixManifest(id)));
                            }
                        }
                    }
                    Some(digests) => {
                        for (index, digest) in digests.iter().enumerate() {
                            if fetch.in_flight.len() >= MATRIX_CHUNKS_IN_FLIGHT {
                                break;
                            }
                            let index = index as u32;
                            if fetch.chunks[index as usize].is_some()
                                || fetch.in_flight.contains_key(&index)
                            {
                                continue;
                            }
                            let Some(peer) = pick(fetch) else {
                                break;
                            };
                            fetch.in_flight.insert(index, (peer, now));
                            issued.push((
                                peer,
                                ClientObjectRequest::MatrixChunk {
                                    file: id,
                                    index,
                                    digest: *digest,
                                },
                            ));
                        }
                    }
                }
            }
            for (peer, request) in issued {
                let token = state.next_token;
                state.next_token += 1;
                state.tokens.insert(token, (peer, request, now));
                out.push(ClientObjectFetch {
                    token,
                    peer,
                    request,
                });
            }
        }
        out
    }

    /// A request of this node failed (`liar`: the peer served false bytes).
    pub fn on_failed(&self, token: u64, liar: bool) -> bool {
        let mut state = self.state.lock().expect("client objects lock");
        let Some((peer, request, _)) = state.tokens.remove(&token) else {
            return false;
        };
        release(&mut state, &peer, &request, liar)
    }

    /// A request of this node was answered with `bytes`, exactly the
    /// requested object (checked by the transport codec, off the event loop:
    /// a chunk is not hashed again here). Called on the event loop: nothing
    /// here is CPU-heavy.
    pub fn on_fetched(
        &self,
        token: u64,
        bytes: &[u8],
    ) -> Result<ClientObjectFetched, ClientObjectsError> {
        let (peer, request) = {
            let mut state = self.state.lock().expect("client objects lock");
            let Some((peer, request, _)) = state.tokens.remove(&token) else {
                return Ok(ClientObjectFetched::Nothing);
            };
            (peer, request)
        };
        match request {
            ClientObjectRequest::Proof(id) => {
                let mut state = self.state.lock().expect("client objects lock");
                if !id.matches_bytes(bytes) {
                    state.proofs.failed(&peer, &id, true);
                    return Ok(ClientObjectFetched::Liar);
                }
                state.proofs.completed(&peer, &id);
                Ok(ClientObjectFetched::Bundle(bytes.to_vec()))
            }
            ClientObjectRequest::MatrixManifest(id) => {
                let mut state = self.state.lock().expect("client objects lock");
                let Some(fetch) = state.fetches.get_mut(&id) else {
                    return Ok(ClientObjectFetched::Nothing);
                };
                fetch.in_flight.remove(&u32::MAX);
                match id.verify_manifest(bytes) {
                    Some(digests) => {
                        fetch.chunks = vec![None; digests.len()];
                        fetch.manifest = Some(digests);
                        Ok(ClientObjectFetched::Nothing)
                    }
                    None => {
                        fetch.excluded.insert(peer);
                        Ok(ClientObjectFetched::Liar)
                    }
                }
            }
            ClientObjectRequest::MatrixChunk {
                file,
                index,
                digest,
            } => {
                let mut state = self.state.lock().expect("client objects lock");
                let Some(fetch) = state.fetches.get_mut(&file) else {
                    return Ok(ClientObjectFetched::Nothing);
                };
                fetch.in_flight.remove(&index);
                let expected = fetch
                    .manifest
                    .as_ref()
                    .and_then(|digests| digests.get(index as usize))
                    .copied();
                // The transport hashed the bytes against the request's digest
                // (the one pass over a chunk, off the event loop): what is
                // left is that the request is the manifest's.
                if expected != Some(digest) || file.chunk_len(index) != Some(bytes.len()) {
                    fetch.excluded.insert(peer);
                    return Ok(ClientObjectFetched::Liar);
                }
                fetch.chunks[index as usize] = Some(bytes.to_vec());
                if !fetch.chunks.iter().all(Option::is_some) {
                    return Ok(ClientObjectFetched::Nothing);
                }
                // Complete: authenticated off the event loop; meanwhile not
                // fetched again.
                let fetch = state.fetches.remove(&file).expect("found above");
                state.authenticating.insert(file);
                tracing::info!(
                    matrix_digest = %hex32(&file.matrix_digest),
                    bytes = file.file_len,
                    chunks = fetch.chunks.len(),
                    "registered client matrix received from peers; authenticating it"
                );
                Ok(ClientObjectFetched::MatrixAssembled(AssembledMatrixFile {
                    id: file,
                    digests: fetch.manifest.expect("chunks follow the manifest"),
                    chunks: fetch.chunks.into_iter().flatten().collect(),
                }))
            }
        }
    }

    /// Receive a client proof bundle (fetched, or handed to this node):
    /// decode it under its bounds, check its client is registered with
    /// `registry`, active at `next_height` (the next block's height: no block
    /// may carry it before) and its matrix held, pre-pass the proof once
    /// (CPU-heavy: run off the reactor), check it is the submission's client,
    /// and queue it. Returns the announcement for peers. Admission and
    /// refusal are logged at INFO, `D` in hex.
    pub fn receive_bundle(
        &self,
        bytes: &[u8],
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        next_height: u64,
    ) -> Result<ClientProofAnnouncement, ClientObjectsError> {
        let outcome = self.admit_bundle(bytes, registry, rules, next_height);
        match &outcome {
            Ok(announcement) => tracing::info!(
                matrix_digest = %hex32(&announcement.id.submission.matrix_digest),
                io_commitment = %hex32(&announcement.id.submission.io_commitment),
                fee = announcement.fee,
                queued = self.queued(),
                "client proof admitted and queued for the miners"
            ),
            Err(error) => tracing::info!(
                matrix_digest = %ClientProofBundle::decode(bytes)
                    .map(|bundle| hex32(&bundle.submission.matrix_digest))
                    .unwrap_or_else(|_| "undecodable".into()),
                %error,
                "client proof refused"
            ),
        }
        outcome
    }

    /// [`Self::receive_bundle`], unlogged.
    fn admit_bundle(
        &self,
        bytes: &[u8],
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        next_height: u64,
    ) -> Result<ClientProofAnnouncement, ClientObjectsError> {
        let bundle =
            ClientProofBundle::decode(bytes).map_err(|error| ClientObjectsError::Bundle(error.to_string()))?;
        let id = ClientProofId::of_bytes(bundle.submission, bytes)
            .ok_or_else(|| ClientObjectsError::Bundle("too long".into()))?;
        if let Some(entry) = registry.entry(&bundle.submission.matrix_digest) {
            if !rules.client_active_at(entry, next_height) {
                return Err(ClientObjectsError::NotYetActive {
                    matrix_digest: hex32(&entry.matrix_digest),
                    tip_height: next_height.saturating_sub(1),
                    active_from: rules.effective_active_from(entry),
                });
            }
        }
        let fee = jetsam_tx::validate_paged_spend(&bundle.payment.pages)
            .map_err(|_| ClientObjectsError::Bundle("payment is not a logical transaction".into()))?
            .fee;
        let prepared = self.prepare_client(&bundle.submission, &bundle.proof_bytes, registry)?;
        let candidate = ClientCandidate {
            submission: bundle.submission,
            payment: bundle.payment,
            payload: Arc::new(prepared),
        };
        {
            let mut state = self.state.lock().expect("client objects lock");
            match state.queue.insert(candidate, rules) {
                Ok(_) => {}
                Err(jetsam_chain::consensus::client_objects::queue::ClientQueueError::Duplicate) => {}
                Err(error) => return Err(ClientObjectsError::Queue(error)),
            }
            state.bundles.insert(id, Arc::from(bytes));
        }
        let path = self
            .proof_dir
            .join(format!("{}.bundle", hex32(&id.bundle_digest)));
        write_atomically(&path, bytes)?;
        Ok(ClientProofAnnouncement { id, fee })
    }

    /// The pre-pass of one client proof under the chain's registry.
    fn prepare_client(
        &self,
        submission: &ClientSubmission,
        proof_bytes: &[u8],
        registry: &ClientRegistryState,
    ) -> Result<PreparedHistoryStepClient, ClientObjectsError> {
        let digest = submission.matrix_digest;
        if registry.entry(&digest).is_none() {
            return Err(ClientObjectsError::UnknownClient(hex32(&digest)));
        }
        // Held, so authenticated once already: the pre-pass reuses that `D`
        // instead of digesting the matrix again.
        let authenticated = self
            .matrices
            .authenticated(&digest)
            .ok_or_else(|| ClientObjectsError::UnknownClient(hex32(&digest)))?;
        let (field_proof, root, io) =
            jetsam_recursive::decode_history_step_client_proof(&self.form, proof_bytes)
                .map_err(|error| ClientObjectsError::Proof(error.to_string()))?;
        if jetsam_recursive::client_io_commitment(&io) != submission.io_commitment {
            return Err(ClientObjectsError::Proof(
                "the proof's public IO is not the submission's".into(),
            ));
        }
        let witness = HistoryStepClientWitness {
            field_proof,
            commitment: jetsam_ivc_core::pcs::Commitment {
                root,
                params: self.form.pcs_params().clone(),
            },
            io,
            matrix: Arc::clone(authenticated.matrix()),
            registry: HistoryStepClientRegistry::new(self.form.registry_depth(), registry.digests())
                .map_err(|error| ClientObjectsError::Proof(error.to_string()))?,
        };
        PreparedHistoryStepClient::prepare_authenticated(&self.form, &witness, &authenticated)
            .map_err(|error| ClientObjectsError::Proof(error.to_string()))
    }

    /// Re-read the bundles kept on disk (after a restart) and receive each
    /// again under `registry`, the next block at `next_height`. Bundles that
    /// are no longer admissible are dropped.
    pub fn reload_bundles(
        &self,
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        next_height: u64,
    ) -> Result<usize, ClientObjectsError> {
        let mut received = 0usize;
        for entry in std::fs::read_dir(&self.proof_dir)? {
            let entry = entry?;
            if entry.path().extension().and_then(|ext| ext.to_str()) != Some("bundle") {
                continue;
            }
            let bytes = read_bounded(
                &entry.path(),
                jetsam_p2p::client_object_protocol::MAX_CLIENT_PROOF_BUNDLE_BYTES as u64,
            )?;
            match self.admit_bundle(&bytes, registry, rules, next_height) {
                Ok(_) => received += 1,
                Err(error) => {
                    tracing::info!(file = %entry.path().display(), %error, "kept client proof dropped on reload");
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        Ok(received)
    }

    /// The client the block after `parent` should carry, if any (the queue's
    /// choice, `anchor_ok` saying whether a payment's epoch anchor is still
    /// minable).
    pub fn best_candidate(
        &self,
        parent: &jetsam_chain::BlockHeader,
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        anchor_ok: impl Fn(&[u8; 32]) -> bool,
    ) -> Option<ClientCandidate<Arc<PreparedHistoryStepClient>>> {
        let state = self.state.lock().expect("client objects lock");
        state
            .queue
            .best(parent, registry, rules, |anchor| anchor_ok(anchor))
            .cloned()
    }

    /// Queued client proofs.
    pub fn queued(&self) -> usize {
        self.state.lock().expect("client objects lock").queue.len()
    }

    /// The registration a matrix file makes: `D` (the structural digest of
    /// the matrix it opens as, under the client form), its root and length.
    pub fn registration_of_matrix_file(
        &self,
        bytes: &[u8],
    ) -> Result<ClientRegistration, ClientObjectsError> {
        Ok(self.open_matrix_file(bytes)?.0)
    }

    /// The registration a matrix file makes, with the matrix it opens as and
    /// its chunk digests: one pass of each hash.
    fn open_matrix_file(
        &self,
        bytes: &[u8],
    ) -> Result<(ClientRegistration, AuthenticatedClientMatrix, Vec<Hash32>), ClientObjectsError>
    {
        if bytes.is_empty() || bytes.len() > CLIENT_MATRIX_MAX_FILE_BYTES as usize {
            return Err(ClientObjectsError::MatrixFile("length out of bounds"));
        }
        let matrix = self
            .open_client_matrix(&mut &bytes[..])
            .ok_or(ClientObjectsError::MatrixFile("not a matrix of the client form"))?;
        let digests = matrix_chunk_digests(self.authentication_priority(), bytes);
        let registration = ClientRegistration {
            matrix_digest: matrix.digest(),
            matrix_file_root: matrix_file_root_from_chunk_digests(bytes.len() as u64, &digests),
            matrix_file_len: bytes.len() as u32,
        };
        Ok((registration, matrix, digests))
    }

    /// Hold the registration `matrix_file` makes, paid by `payment`, for this
    /// node's miner: the payment must open exactly this registration and pay
    /// its license (`check_registration_payment`); the matrix is then held
    /// and served, so the network can fetch it during the activation delay.
    /// The caller checks what depends on the chain (state of the payment,
    /// registry not full, `D` not registered yet). Logged at INFO, `D` in
    /// hex: held, or refused and why.
    pub fn hold_client_registration(
        &self,
        payment: PagedSpendIntent,
        matrix_file: &[u8],
        rules: &ClientObjectRules,
    ) -> Result<ClientRegistration, ClientObjectsError> {
        let (registration, matrix, digests) = match self.open_matrix_file(matrix_file) {
            Ok(opened) => opened,
            Err(error) => {
                tracing::info!(
                    bytes = matrix_file.len(),
                    %error,
                    "client registration refused: the file is not a client matrix"
                );
                return Err(error);
            }
        };
        let digest = hex32(&registration.matrix_digest);
        match self.hold_opened_registration(payment, matrix_file, rules, registration, matrix, digests)
        {
            Ok(registration) => {
                tracing::info!(
                    matrix_digest = %digest,
                    matrix_file_root = %hex32(&registration.matrix_file_root),
                    bytes = registration.matrix_file_len,
                    "client registration accepted: held for the miners, its matrix served to peers"
                );
                Ok(registration)
            }
            Err(error) => {
                tracing::info!(matrix_digest = %digest, %error, "client registration refused");
                Err(error)
            }
        }
    }

    fn hold_opened_registration(
        &self,
        payment: PagedSpendIntent,
        matrix_file: &[u8],
        rules: &ClientObjectRules,
        registration: ClientRegistration,
        matrix: AuthenticatedClientMatrix,
        digests: Vec<Hash32>,
    ) -> Result<ClientRegistration, ClientObjectsError> {
        let pages: Vec<jetsam_tx::Transaction> = payment
            .pages
            .iter()
            .map(|page| jetsam_tx::Transaction::new(page.body.clone()))
            .collect();
        jetsam_chain::consensus::client_objects::check_registration_payment(
            &pages,
            &registration,
            rules,
        )
        .map_err(|error| ClientObjectsError::Registration(error.to_string()))?;
        let id = MatrixFileId::of_registration(&registration);
        let path = self.matrix_dir.join(matrix_file_name(&id));
        write_atomically(&path, matrix_file)?;
        self.hold_authenticated(id, matrix, digests, path)?;
        self.hold_registration(HeldRegistration {
            registration,
            payment,
        })?;
        Ok(registration)
    }

    /// Hold a registration relayed by a peer (§8 gap b), its payment already
    /// checked to open and pay it (by the transport) and against this node's
    /// state (by the caller, through the mempool): refused if `D` is already
    /// registered with `registry` or the registry is full. Returns whether it
    /// is newly held (a registration already held keeps its first payment).
    pub fn hold_relayed_registration(
        &self,
        notice: &jetsam_p2p::client_object_protocol::ClientRegistrationNotice,
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
    ) -> Result<bool, ClientObjectsError> {
        let digest = notice.registration.matrix_digest;
        if registry.entry(&digest).is_some() {
            return Err(ClientObjectsError::Registration(
                "this client is already registered".into(),
            ));
        }
        if registry.len()
            >= rules
                .registry_capacity
                .min(jetsam_chain::consensus::client_objects::CLIENT_REGISTRY_CAPACITY)
        {
            return Err(ClientObjectsError::Registration(
                "the client registry is full".into(),
            ));
        }
        if self
            .state
            .lock()
            .expect("client objects lock")
            .registrations
            .contains_key(&digest)
        {
            return Ok(false);
        }
        self.hold_registration(HeldRegistration {
            registration: notice.registration,
            payment: notice.payment.clone(),
        })?;
        Ok(true)
    }

    /// Hold a registration for this node's miner.
    pub fn hold_registration(&self, held: HeldRegistration) -> Result<(), ClientObjectsError> {
        let mut state = self.state.lock().expect("client objects lock");
        if state.registrations.len() >= MAX_HELD_REGISTRATIONS
            && !state
                .registrations
                .contains_key(&held.registration.matrix_digest)
        {
            return Err(ClientObjectsError::Registration(
                "too many registrations held".into(),
            ));
        }
        // Kept on disk as its notice bytes (the relay's encoding). A payment
        // too large to relay is held in memory only.
        let digest = held.registration.matrix_digest;
        let notice = jetsam_p2p::client_object_protocol::ClientRegistrationNotice {
            registration: held.registration,
            payment: held.payment.clone(),
        };
        if notice.encode().len()
            <= jetsam_p2p::client_object_protocol::MAX_CLIENT_REGISTRATION_NOTICE_BYTES
        {
            write_atomically(
                &self.registration_dir.join(registration_file_name(&digest)),
                &notice.encode(),
            )?;
        }
        state.registrations.insert(digest, held);
        Ok(())
    }

    /// Registrations held for this node's miner, not yet in `registry`.
    pub fn held_registrations(&self, registry: &ClientRegistryState) -> Vec<HeldRegistration> {
        self.state
            .lock()
            .expect("client objects lock")
            .registrations
            .values()
            .filter(|held| registry.entry(&held.registration.matrix_digest).is_none())
            .cloned()
            .collect()
    }

    /// Forget what a committed block settles: the submission it paid, every
    /// candidate whose payment spends an input it spent, and the
    /// registrations it made.
    pub fn on_block_committed(&self, block: &jetsam_chain::Block) {
        let mut state = self.state.lock().expect("client objects lock");
        state.queue.on_block_committed(block);
        for object in &block.client_objects {
            match object {
                ClientObject::Registration(registration) => {
                    state.registrations.remove(&registration.matrix_digest);
                    let _ = std::fs::remove_file(
                        self.registration_dir
                            .join(registration_file_name(&registration.matrix_digest)),
                    );
                }
                ClientObject::Submission(submission) => {
                    let ids: Vec<ClientProofId> = state
                        .bundles
                        .keys()
                        .filter(|id| id.submission == *submission)
                        .copied()
                        .collect();
                    for id in ids {
                        state.bundles.remove(&id);
                        let _ = std::fs::remove_file(
                            self.proof_dir
                                .join(format!("{}.bundle", hex32(&id.bundle_digest))),
                        );
                    }
                }
            }
        }
    }

    /// Matrix files held, for listings.
    pub fn held_matrix_files(&self) -> Vec<MatrixFileId> {
        self.state
            .lock()
            .expect("client objects lock")
            .held_files
            .keys()
            .copied()
            .collect()
    }
}

fn release(state: &mut State, peer: &PeerId, request: &ClientObjectRequest, liar: bool) -> bool {
    match request {
        ClientObjectRequest::Proof(id) => {
            if liar {
                state.proofs.failed(peer, id, true)
            } else {
                state.proofs.unavailable(peer, id)
            }
        }
        ClientObjectRequest::MatrixManifest(file) => {
            if let Some(fetch) = state.fetches.get_mut(file) {
                fetch.in_flight.remove(&u32::MAX);
                if liar {
                    fetch.excluded.insert(*peer);
                }
            }
            liar
        }
        ClientObjectRequest::MatrixChunk { file, index, .. } => {
            if let Some(fetch) = state.fetches.get_mut(file) {
                fetch.in_flight.remove(index);
                if liar {
                    fetch.excluded.insert(*peer);
                }
            }
            liar
        }
    }
}

fn read_bounded(path: &Path, max: u64) -> std::io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    Read::by_ref(&mut file).take(max + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > max {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "client object file exceeds its bound",
        ));
    }
    Ok(bytes)
}

impl jetsam_miner::client_slot::MinerClientSource for ClientObjects {
    fn best_candidate(
        &self,
        parent: &jetsam_chain::BlockHeader,
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    ) -> Option<ClientCandidate<Arc<PreparedHistoryStepClient>>> {
        ClientObjects::best_candidate(self, parent, registry, rules, anchor_ok)
    }

    fn held_registrations(
        &self,
        registry: &ClientRegistryState,
    ) -> Vec<(ClientRegistration, PagedSpendIntent)> {
        ClientObjects::held_registrations(self, registry)
            .into_iter()
            .map(|held| (held.registration, held.payment))
            .collect()
    }

    fn on_block_committed(&self, block: &jetsam_chain::Block) {
        ClientObjects::on_block_committed(self, block);
    }
}

impl jetsam_rpc::client_objects::RpcClientObjects for ClientObjects {
    fn receive_client_proof(
        &self,
        bundle: &[u8],
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        next_height: u64,
    ) -> Result<ClientProofAnnouncement, String> {
        self.receive_bundle(bundle, registry, rules, next_height)
            .map_err(|error| error.to_string())
    }

    fn hold_client_registration(
        &self,
        payment: PagedSpendIntent,
        matrix_file: &[u8],
        rules: &ClientObjectRules,
    ) -> Result<ClientRegistration, String> {
        ClientObjects::hold_client_registration(self, payment, matrix_file, rules)
            .map_err(|error| error.to_string())
    }

    fn registration_of_matrix_file(&self, matrix_file: &[u8]) -> Result<ClientRegistration, String> {
        ClientObjects::registration_of_matrix_file(self, matrix_file)
            .map_err(|error| error.to_string())
    }

    fn holds_client_matrix(&self, matrix_digest: &[u8; 32]) -> bool {
        self.holds_matrix(matrix_digest)
    }

    fn queued_client_proofs(&self) -> usize {
        self.queued()
    }
}

impl ClientObjectSource for ClientObjects {
    fn client_object(&self, request: &ClientObjectRequest) -> Option<Vec<u8>> {
        match request {
            ClientObjectRequest::Proof(id) => self
                .state
                .lock()
                .expect("client objects lock")
                .bundles
                .get(id)
                .map(|bytes| bytes.to_vec()),
            ClientObjectRequest::MatrixManifest(file) => self
                .state
                .lock()
                .expect("client objects lock")
                .manifests
                .get(file)
                .cloned(),
            ClientObjectRequest::MatrixChunk { file, index, .. } => {
                let path = self
                    .state
                    .lock()
                    .expect("client objects lock")
                    .held_files
                    .get(file)
                    .cloned()?;
                let mut handle = std::fs::File::open(path).ok()?;
                let offset = u64::from(*index) * CLIENT_MATRIX_CHUNK_BYTES as u64;
                let len = file.chunk_len(*index)?;
                std::io::Seek::seek(&mut handle, std::io::SeekFrom::Start(offset)).ok()?;
                let mut chunk = vec![0u8; len];
                handle.read_exact(&mut chunk).ok()?;
                Some(chunk)
            }
        }
    }
}

#[cfg(test)]
mod tests;
