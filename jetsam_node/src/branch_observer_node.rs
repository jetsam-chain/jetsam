// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The branch observer in the daemon (plan T3.10b): what
//! [`crate::branch_observer`] reads from the node, and when.
//!
//! * [`StoreCanonicalView`]: the canonical chain as the store holds it (tip,
//!   headers, cumulative work), read where the observer runs.
//! * [`SharedBranchObserver`]: the one record of the event loop, the RPC
//!   (`getObservedBranches`) and shutdown. It is read back at startup — a
//!   damaged or unreadable file is reported and a new record starts — and
//!   written at shutdown and at most every minute.
//! * [`BranchProbes`]: the event loop's side. Each heartbeat lists the
//!   refused peers to the observer, sends the probes due on free header
//!   lanes, and hands the P2P layer the peers whose probe is not over
//!   (`NetworkCommand::ProbingPeers`), whenever that set or the refused set
//!   it depends on changed: the P2P layer keeps a peer for a probe only while
//!   that peer is on its refused list. The answer to a probe and an accepted
//!   reorganization are read off the event loop, on a blocking worker:
//!   weighing a branch checks the proof of work of each new header (1.5 to
//!   4.7 ms each, `jetsam_chain::consensus::pow`), up to 512 per answer. A
//!   heartbeat that finds the observer busy does nothing that round.
//!
//! No tip is known of a refused peer: the node keeps per-peer tips only for
//! the peers it agrees with (`MiningPeerQuorum`), its highest announced
//! height is global, and a refused peer's manifest is refused before it is
//! read. The probes are therefore sent without an announced tip
//! (`DivergentPeer::announced_tip = None`): the climb goes as far as the peer
//! serves, then asks again every [`crate::branch_observer::PROBE_DELAY_SECS`],
//! within the 10 080 headers of a branch and the P2P layer's 10 minutes.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, TryLockError};

use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::storage::MdbxStore;
use jetsam_p2p::header_protocol::HeaderInventoryRecord;
use jetsam_rpc::observed_branches::{ObservedBranchesResponse, ObservedBranchesSource};
use libp2p::PeerId;

use crate::branch_observer::{BranchObserver, BranchProbe, CanonicalView, DivergentPeer};
use crate::networking::ChainPoint;

/// The canonical chain as `store` holds it, its tip read when the view is
/// made.
#[derive(Clone)]
pub struct StoreCanonicalView {
    store: MdbxStore,
    tip: ChainPoint,
}

impl StoreCanonicalView {
    /// The canonical chain `store` holds now; `None` when it holds no tip.
    pub fn at_tip(store: MdbxStore) -> Option<Self> {
        let (height, hash) = store.get_chain_tip().ok().flatten()?;
        Some(Self {
            store,
            tip: ChainPoint::new(height, hash),
        })
    }
}

impl CanonicalView for StoreCanonicalView {
    fn tip(&self) -> ChainPoint {
        self.tip
    }

    fn header_at(&self, height: u64) -> Option<BlockHeader> {
        self.store.get_header(height).ok().flatten()
    }

    fn work_at(&self, height: u64) -> Option<[u8; 32]> {
        self.store.get_chain_work(height).ok().flatten()
    }
}

/// The node's one record of refused branches, and its file.
#[derive(Clone)]
pub struct SharedBranchObserver {
    observer: Arc<Mutex<BranchObserver>>,
    path: PathBuf,
}

impl SharedBranchObserver {
    /// The record the last run saved at `path`; a new one when there is no
    /// file, or when it is damaged or unreadable (reported).
    pub fn load(path: PathBuf, now_unix: u64) -> Self {
        let observer = match BranchObserver::load(&path, now_unix) {
            Ok(observer) => observer,
            Err(error) => {
                tracing::warn!(
                    %error,
                    path = %path.display(),
                    "the branch observer's record of the last run is unreadable: a new record starts"
                );
                BranchObserver::new(now_unix)
            }
        };
        Self {
            observer: Arc::new(Mutex::new(observer)),
            path,
        }
    }

    /// Write the record to its file (shutdown).
    pub fn save(&self, now_unix: u64) -> std::io::Result<()> {
        self.lock().save(&self.path, now_unix)
    }

    /// The answer of `from` to its probe, against `canonical`. It waits for
    /// the observer and weighs headers: never on the event loop.
    pub fn read_answer(
        &self,
        from: PeerId,
        records: &[HeaderInventoryRecord],
        canonical: &impl CanonicalView,
        now_unix: u64,
    ) {
        self.lock().on_headers(from, records, canonical, now_unix);
    }

    /// This node reorganized onto the canonical chain above `fork_height`,
    /// undoing `depth` blocks. Never on the event loop.
    pub fn record_reorg(
        &self,
        fork_height: u64,
        depth: u64,
        canonical: &impl CanonicalView,
        now_unix: u64,
    ) {
        self.lock()
            .on_accepted_reorg(fork_height, depth, canonical, now_unix);
    }

    fn lock(&self) -> MutexGuard<'_, BranchObserver> {
        self.observer.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl ObservedBranchesSource for SharedBranchObserver {
    fn observed_branches(
        &self,
        since_unix: u64,
        canonical_tip: ChainPoint,
        node_tip_work: [u8; 32],
        now_unix: u64,
    ) -> ObservedBranchesResponse {
        self.lock()
            .observed_branches(since_unix, canonical_tip, node_tip_work, now_unix)
    }
}

/// The event loop's side of the observer (module comment).
pub struct BranchProbes {
    shared: SharedBranchObserver,
    /// Peers sent a probe whose next header batch has not come yet.
    awaited: HashSet<PeerId>,
    /// The probing set last handed to the P2P layer, and the refused set the
    /// P2P layer held then.
    handed: Option<(HashSet<PeerId>, HashSet<PeerId>)>,
}

impl BranchProbes {
    pub fn new(shared: SharedBranchObserver) -> Self {
        Self {
            shared,
            awaited: HashSet::new(),
            handed: None,
        }
    }

    /// One heartbeat, unless the observer is busy (`false`: nothing done).
    /// `divergent` are the refused peers with their divergence heights;
    /// `refused_handed`, the refused set the P2P layer holds. Each probe due
    /// to a peer of `connected` whose header lane is free (not in `busy`) goes
    /// through `dispatch` (true when sent); the peers whose probe is not over
    /// go through `hand_probing` (true when handed) when they, or
    /// `refused_handed`, changed since they were last handed. The record is
    /// written at most every minute.
    #[allow(clippy::too_many_arguments)]
    pub fn heartbeat(
        &mut self,
        divergent: impl IntoIterator<Item = (PeerId, u64)>,
        refused_handed: &HashSet<PeerId>,
        connected: &HashSet<PeerId>,
        busy: &HashSet<PeerId>,
        now_unix: u64,
        mut dispatch: impl FnMut(&BranchProbe) -> bool,
        hand_probing: impl FnOnce(Vec<PeerId>) -> bool,
    ) -> bool {
        let mut observer = match self.shared.observer.try_lock() {
            Ok(observer) => observer,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return false,
        };
        let divergent = divergent
            .into_iter()
            .map(|(peer, divergence_height)| DivergentPeer {
                peer,
                divergence_height,
                announced_tip: None,
            })
            .collect::<Vec<_>>();
        observer.on_divergent_peers(&divergent, now_unix);
        for probe in observer.probes_due(connected, busy, now_unix) {
            if dispatch(&probe) {
                self.awaited.insert(probe.peer);
            }
        }
        let probing = observer.probing_peers();
        self.awaited.retain(|peer| probing.contains(peer));
        let unchanged = self
            .handed
            .as_ref()
            .is_some_and(|(handed, refused)| *handed == probing && refused == refused_handed);
        if !unchanged {
            let mut peers = probing.iter().copied().collect::<Vec<_>>();
            peers.sort_by_key(|peer| peer.to_bytes());
            if hand_probing(peers) {
                self.handed = Some((probing, refused_handed.clone()));
            }
        }
        if let Err(error) = observer.save_if_due(&self.shared.path, now_unix) {
            tracing::warn!(%error, "the branch observer's record could not be saved");
        }
        true
    }

    /// Whether a header batch from `from` may answer a probe: once per probe
    /// sent, for the first batch from `from` since.
    pub fn takes_answer(&mut self, from: &PeerId) -> bool {
        self.awaited.remove(from)
    }

    /// A header batch from `from`. When it is the first since a probe was
    /// sent to `from`, the observer reads it, against `store`'s canonical
    /// chain, on a blocking worker (the handle is returned); the batch goes on
    /// to the node's own handling unchanged either way.
    pub fn observe_answer(
        &mut self,
        from: PeerId,
        records: &[HeaderInventoryRecord],
        store: &MdbxStore,
    ) -> Option<tokio::task::JoinHandle<()>> {
        if !self.takes_answer(&from) {
            return None;
        }
        let (shared, records, store) = (self.shared.clone(), records.to_vec(), store.clone());
        Some(tokio::task::spawn_blocking(move || {
            if let Some(canonical) = StoreCanonicalView::at_tip(store) {
                shared.read_answer(from, &records, &canonical, unix_now());
            }
        }))
    }

    /// An accepted reorganization, from the heights the chain reverted: read
    /// by the observer against `store`'s canonical chain, on a blocking
    /// worker (the handle is returned).
    pub fn observe_reorg(
        &self,
        reverted: &[u64],
        store: &MdbxStore,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let (fork, depth) = accepted_reorg_fork(reverted)?;
        let (shared, store) = (self.shared.clone(), store.clone());
        Some(tokio::task::spawn_blocking(move || {
            if let Some(canonical) = StoreCanonicalView::at_tip(store) {
                shared.record_reorg(fork, depth, &canonical, unix_now());
            }
        }))
    }
}

/// `(fork, depth)` of an accepted reorganization from the heights it
/// reverted: the chain reverts `fork + 1 ..= old tip` and applies
/// `fork + 1 ..= new tip` (`MdbxChainContext::apply_reorg_mdbx_with_objects`).
/// `None` when nothing was reverted.
pub fn accepted_reorg_fork(reverted: &[u64]) -> Option<(u64, u64)> {
    let lowest = reverted.iter().min()?;
    Some((lowest.checked_sub(1)?, reverted.len() as u64))
}

/// Seconds since the Unix epoch.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests;
