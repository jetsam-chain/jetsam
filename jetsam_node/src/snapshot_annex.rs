// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T7.1): the annexes of a snapshot at or above J between nodes.
//!
//! A boundary at or above J carries two files beside its UTXO state — the
//! destination state file and the guardian journal — too large for the
//! manifest's control frame. The manifest commits to each by kind, exact
//! length and root; the files move on the client-object protocol as
//! content-addressed files (the pattern of the v1.5 matrix files): the index
//! (the chunk digests, checked against the root), then 1 MiB chunks, each
//! checked against its digest by the requester's codec before anything reads
//! it.
//!
//! - **Serving** ([`NodeObjectSource`]): the annexes of this node's own
//!   snapshot generations, found by reference in the export directory, each
//!   chunk read from disk and checked again before it leaves; everything
//!   else is the client objects' ([`ClientObjects`]).
//! - **Fetching** ([`fetch_snapshot_annexes`]): after the manifest's
//!   admission and before the install, from the peers of the snapshot plan;
//!   a peer that serves other bytes is excluded and named. The bytes are
//!   believed for nothing: the install authenticates them against the
//!   installed state and headers.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use jetsam_chain::storage::{
    open_snapshot_generation, SnapshotAnnexError, SnapshotAnnexKind, SnapshotAnnexRef,
    SnapshotAnnexes, SnapshotGeneration,
};
use jetsam_p2p::client_object_codec::ClientObjectRequest;
use jetsam_p2p::client_object_transport::ClientObjectSource;
use libp2p::PeerId;
use tokio::sync::oneshot;

use crate::client_objects::ClientObjects;

/// Tokens of annex fetches carry this bit; the client objects' never do
/// (their counter starts at zero and never reaches it).
pub const SNAPSHOT_ANNEX_TOKEN_BIT: u64 = 1 << 63;

/// How long one annex request may stay unanswered (the transport's own
/// deadline is 75 s).
pub const SNAPSHOT_ANNEX_REQUEST_TIMEOUT: Duration = Duration::from_secs(90);

/// Whether a client-object token is an annex fetch's.
pub fn is_snapshot_annex_token(token: u64) -> bool {
    token & SNAPSHOT_ANNEX_TOKEN_BIT != 0
}

// ---------------------------------------------------------------------------
// Serving
// ---------------------------------------------------------------------------

/// What serves annexes: the generations of an export directory.
trait AnnexFiles: Send + Sync {
    fn index(&self, annex: &SnapshotAnnexRef) -> Option<Vec<u8>>;
    fn chunk(&self, annex: &SnapshotAnnexRef, index: u32) -> Option<Vec<u8>>;
}

impl AnnexFiles for SnapshotGeneration {
    fn index(&self, annex: &SnapshotAnnexRef) -> Option<Vec<u8>> {
        self.annex_index(annex)
    }

    fn chunk(&self, annex: &SnapshotAnnexRef, index: u32) -> Option<Vec<u8>> {
        self.read_annex_chunk(annex, index).ok()
    }
}

/// The annexes of this node's snapshot generations, by reference. A
/// generation directory is immutable and content-named: each is opened (its
/// annexes read and digested) once, then served chunk by chunk from disk.
pub struct SnapshotAnnexSource {
    export_root: PathBuf,
    generations: Mutex<BTreeMap<PathBuf, Arc<dyn AnnexFiles>>>,
}

impl SnapshotAnnexSource {
    pub fn new(export_root: impl AsRef<Path>) -> Self {
        Self {
            export_root: export_root.as_ref().to_path_buf(),
            generations: Mutex::new(BTreeMap::new()),
        }
    }

    /// The bytes of an annex request, if this node holds that annex.
    pub fn serve(&self, request: &ClientObjectRequest) -> Option<Vec<u8>> {
        let (annex, chunk) = match request {
            ClientObjectRequest::SnapshotAnnexIndex(annex) => (annex, None),
            ClientObjectRequest::SnapshotAnnexChunk { annex, index, .. } => (annex, Some(*index)),
            _ => return None,
        };
        let bytes = self.find(annex, |files| match chunk {
            None => files.index(annex),
            Some(index) => files.chunk(annex, index),
        })?;
        // Serve only what is exactly the requested object.
        request.matches(&bytes).then_some(bytes)
    }

    fn find(
        &self,
        annex: &SnapshotAnnexRef,
        read: impl Fn(&dyn AnnexFiles) -> Option<Vec<u8>>,
    ) -> Option<Vec<u8>> {
        let lookup = |generations: &BTreeMap<PathBuf, Arc<dyn AnnexFiles>>| {
            generations.values().find_map(|files| {
                files.index(annex)?;
                read(files.as_ref())
            })
        };
        if let Some(bytes) = lookup(&self.generations.lock().expect("annex source lock")) {
            return Some(bytes);
        }
        self.refresh();
        lookup(&self.generations.lock().expect("annex source lock"))
    }

    /// Forget the generations that disappeared, open the new ones.
    fn refresh(&self) {
        let present: Vec<PathBuf> = std::fs::read_dir(&self.export_root)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
                    .map(|entry| entry.path())
                    .collect()
            })
            .unwrap_or_default();
        let known: HashSet<PathBuf> = {
            let mut generations = self.generations.lock().expect("annex source lock");
            generations.retain(|path, _| present.contains(path));
            generations.keys().cloned().collect()
        };
        for path in present {
            if known.contains(&path) {
                continue;
            }
            if let Ok(generation) = open_snapshot_generation(&path) {
                if !generation.annexes().is_empty() {
                    self.generations
                        .lock()
                        .expect("annex source lock")
                        .insert(path, Arc::new(generation));
                }
            }
        }
    }

    #[cfg(test)]
    fn insert(&self, path: &str, files: Arc<dyn AnnexFiles>) {
        self.generations
            .lock()
            .expect("annex source lock")
            .insert(PathBuf::from(path), files);
    }
}

/// What this node serves on the client-object protocol: its client objects
/// (v1.5) and the annexes of its snapshot generations (v1.7).
pub struct NodeObjectSource {
    objects: Option<Arc<ClientObjects>>,
    annexes: SnapshotAnnexSource,
}

impl NodeObjectSource {
    pub fn new(objects: Option<Arc<ClientObjects>>, export_root: impl AsRef<Path>) -> Self {
        Self {
            objects,
            annexes: SnapshotAnnexSource::new(export_root),
        }
    }
}

impl ClientObjectSource for NodeObjectSource {
    fn client_object(&self, request: &ClientObjectRequest) -> Option<Vec<u8>> {
        match request {
            ClientObjectRequest::SnapshotAnnexIndex(_)
            | ClientObjectRequest::SnapshotAnnexChunk { .. } => self.annexes.serve(request),
            _ => self.objects.as_ref()?.client_object(request),
        }
    }
}

/// The node's client-object source: as before v1.7 when it serves client
/// objects, or when an armed v1.7 clock makes it serve snapshot annexes;
/// none otherwise (a node of a dormant clock without the v1.5 relation
/// serves nothing, as before).
pub fn node_object_source(
    objects: Option<Arc<ClientObjects>>,
    export_root: &Path,
) -> Option<Arc<dyn ClientObjectSource>> {
    let v1_7_armed = jetsam_chain::consensus::destination::state::DestinationRules::current()
        .activation_height
        .is_some();
    (objects.is_some() || v1_7_armed).then(|| {
        Arc::new(NodeObjectSource::new(objects, export_root)) as Arc<dyn ClientObjectSource>
    })
}

// ---------------------------------------------------------------------------
// Fetching
// ---------------------------------------------------------------------------

/// The answer to one annex request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnnexReply {
    /// The object's bytes (the codec checked them already).
    Fetched(Vec<u8>),
    /// No object: `liar` when the peer served other bytes than the object.
    Failed { liar: bool },
}

/// Hands the transport's answers to annex tokens back to the waiting fetch.
#[derive(Clone, Default)]
pub struct SnapshotAnnexRouter {
    inner: Arc<Mutex<RouterState>>,
}

#[derive(Default)]
struct RouterState {
    next: u64,
    waiting: HashMap<u64, oneshot::Sender<AnnexReply>>,
}

impl SnapshotAnnexRouter {
    fn register(&self) -> (u64, oneshot::Receiver<AnnexReply>) {
        let mut state = self.inner.lock().expect("annex router lock");
        let token = SNAPSHOT_ANNEX_TOKEN_BIT | (state.next & !SNAPSHOT_ANNEX_TOKEN_BIT);
        state.next = state.next.wrapping_add(1);
        let (sender, receiver) = oneshot::channel();
        state.waiting.insert(token, sender);
        (token, receiver)
    }

    fn forget(&self, token: u64) {
        self.inner
            .lock()
            .expect("annex router lock")
            .waiting
            .remove(&token);
    }

    fn deliver(&self, token: u64, reply: AnnexReply) -> bool {
        let sender = self
            .inner
            .lock()
            .expect("annex router lock")
            .waiting
            .remove(&token);
        sender.is_some_and(|sender| sender.send(reply).is_ok())
    }

    /// A `ClientObjectFetched` for an annex token.
    pub fn fetched(&self, token: u64, bytes: &[u8]) -> bool {
        self.deliver(token, AnnexReply::Fetched(bytes.to_vec()))
    }

    /// A `ClientObjectFetchFailed` for an annex token.
    pub fn failed(&self, token: u64, liar: bool) -> bool {
        self.deliver(token, AnnexReply::Failed { liar })
    }
}

/// One annex request at a time to one peer through the node's network.
pub struct NetworkAnnexFetcher {
    commands: jetsam_p2p::NetworkCommandSender,
    router: SnapshotAnnexRouter,
}

impl NetworkAnnexFetcher {
    pub fn new(commands: jetsam_p2p::NetworkCommandSender, router: SnapshotAnnexRouter) -> Self {
        Self { commands, router }
    }

    pub async fn fetch(&self, peer: PeerId, request: ClientObjectRequest) -> AnnexReply {
        let (token, reply) = self.router.register();
        let mut command = jetsam_p2p::NetworkCommand::FetchClientObject {
            token,
            peer,
            request,
        };
        // A full lane is retried briefly; the event loop never awaits it.
        let mut sent = false;
        for _ in 0..50 {
            match self.commands.try_send(command) {
                Ok(()) => {
                    sent = true;
                    break;
                }
                Err(tokio::sync::mpsc::error::TrySendError::Full(back)) => {
                    command = back;
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => break,
            }
        }
        if !sent {
            self.router.forget(token);
            return AnnexReply::Failed { liar: false };
        }
        match tokio::time::timeout(SNAPSHOT_ANNEX_REQUEST_TIMEOUT, reply).await {
            Ok(Ok(reply)) => reply,
            _ => {
                self.router.forget(token);
                AnnexReply::Failed { liar: false }
            }
        }
    }
}

/// Why the annexes of a manifest could not be fetched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AnnexFetchError {
    /// No peer of the plan served this object; `liars` served other bytes.
    Unavailable {
        kind: SnapshotAnnexKind,
        liars: Vec<PeerId>,
    },
}

impl std::fmt::Display for AnnexFetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable { kind, liars } if liars.is_empty() => {
                write!(f, "snapshot annex {kind:?} unavailable from the plan's peers")
            }
            Self::Unavailable { kind, liars } => write!(
                f,
                "snapshot annex {kind:?}: {} peer(s) served other bytes, none the annex",
                liars.len()
            ),
        }
    }
}

/// The annex files of a manifest, as received, and the peers that served
/// other bytes on the way (excluded).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FetchedAnnexes {
    pub files: Vec<(SnapshotAnnexKind, Vec<u8>)>,
    pub liars: Vec<PeerId>,
}

impl FetchedAnnexes {
    /// The decoded annexes the install reads (a file that does not decode is
    /// the sender's).
    pub fn decode(&self) -> Result<SnapshotAnnexes, SnapshotAnnexError> {
        let file = |kind| {
            self.files
                .iter()
                .find(|(held, _)| *held == kind)
                .map(|(_, bytes)| bytes.as_slice())
        };
        SnapshotAnnexes::decode(
            file(SnapshotAnnexKind::DestinationState),
            file(SnapshotAnnexKind::GuardianJournal),
        )
    }
}

/// One object from the first peer that serves exactly it, trying `peers` in
/// order from `start`, skipping the excluded ones; a peer that serves other
/// bytes is excluded (and named).
async fn fetch_object<F, Fut>(
    fetch: &F,
    peers: &[PeerId],
    start: usize,
    request: ClientObjectRequest,
    excluded: &mut HashSet<PeerId>,
    liars: &mut Vec<PeerId>,
) -> Option<Vec<u8>>
where
    F: Fn(PeerId, ClientObjectRequest) -> Fut,
    Fut: Future<Output = AnnexReply>,
{
    for offset in 0..peers.len() {
        let peer = peers[(start + offset) % peers.len()];
        if excluded.contains(&peer) {
            continue;
        }
        match fetch(peer, request).await {
            AnnexReply::Fetched(bytes) if request.matches(&bytes) => return Some(bytes),
            AnnexReply::Fetched(_) | AnnexReply::Failed { liar: true } => {
                excluded.insert(peer);
                liars.push(peer);
            }
            AnnexReply::Failed { liar: false } => {}
        }
    }
    None
}

/// Fetch every annex a manifest commits to, by content, from the plan's
/// `peers`: each annex's index (checked against its root), then each chunk
/// (checked against its digest), spread over the peers. The file grows with
/// the chunks received; it is bounded by the committed length, itself bounded
/// by the kind.
pub async fn fetch_snapshot_annexes<F, Fut>(
    fetch: F,
    peers: &[PeerId],
    annexes: &[SnapshotAnnexRef],
) -> Result<FetchedAnnexes, AnnexFetchError>
where
    F: Fn(PeerId, ClientObjectRequest) -> Fut,
    Fut: Future<Output = AnnexReply>,
{
    let mut excluded = HashSet::new();
    let mut fetched = FetchedAnnexes::default();
    for annex in annexes {
        let unavailable = |liars: &Vec<PeerId>| AnnexFetchError::Unavailable {
            kind: annex.kind,
            liars: liars.clone(),
        };
        if !annex.is_admissible() {
            return Err(unavailable(&fetched.liars));
        }
        let index = fetch_object(
            &fetch,
            peers,
            0,
            ClientObjectRequest::SnapshotAnnexIndex(*annex),
            &mut excluded,
            &mut fetched.liars,
        )
        .await
        .ok_or_else(|| unavailable(&fetched.liars))?;
        let digests = annex
            .verify_index(&index)
            .ok_or_else(|| unavailable(&fetched.liars))?;
        let mut file = Vec::new();
        for (i, digest) in digests.iter().enumerate() {
            let chunk = fetch_object(
                &fetch,
                peers,
                i,
                ClientObjectRequest::SnapshotAnnexChunk {
                    annex: *annex,
                    index: i as u32,
                    digest: *digest,
                },
                &mut excluded,
                &mut fetched.liars,
            )
            .await
            .ok_or_else(|| unavailable(&fetched.liars))?;
            file.extend_from_slice(&chunk);
        }
        fetched.files.push((annex.kind, file));
    }
    Ok(fetched)
}

/// The annexes of an admitted manifest, fetched from the plan's `peers` and
/// decoded, as the install reads them; an error is the plan's (the install
/// is then refused and the plan retired), with the peers that served other
/// bytes logged.
pub async fn fetch_manifest_annexes(
    fetcher: &NetworkAnnexFetcher,
    peers: &[PeerId],
    annexes: &[SnapshotAnnexRef],
) -> Result<SnapshotAnnexes, String> {
    if annexes.is_empty() {
        return Ok(SnapshotAnnexes::none());
    }
    let fetched = fetch_snapshot_annexes(
        |peer, request| fetcher.fetch(peer, request),
        peers,
        annexes,
    )
    .await;
    let liars = match &fetched {
        Ok(fetched) => fetched.liars.clone(),
        Err(AnnexFetchError::Unavailable { liars, .. }) => liars.clone(),
    };
    for liar in liars {
        tracing::warn!(peer = %liar, "peer served bytes failing a snapshot annex's identity");
    }
    let fetched = fetched.map_err(|error| error.to_string())?;
    tracing::info!(
        annexes = fetched.files.len(),
        bytes = fetched.files.iter().map(|(_, bytes)| bytes.len()).sum::<usize>(),
        "snapshot annexes received by content"
    );
    fetched
        .decode()
        .map_err(|error| format!("snapshot annex refused: {error}"))
}

#[cfg(test)]
mod tests;
