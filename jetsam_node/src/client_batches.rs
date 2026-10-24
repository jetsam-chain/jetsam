// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The node's v1.6 batches: batch bundles received over RPC or
//! fetched from a peer, decided once, held for this node's miner and served
//! to peers.
//!
//! - **Decided once, on reception.** A bundle (batch submission, paying
//!   transaction, batch proof) is decoded under its bounds, its class must be
//!   the chain's (`CLIENT_BATCH_CLASS`), and its proof is decided natively
//!   against this node's tip: the registry leaves the tip publishes, the
//!   lanes it publishes (dead at the first v1.6 block) and the form-Z
//!   matrices of the entries it uses. The decision is the batch's
//!   statement — its root and count must be the submission's — and every
//!   entry it uses must be carriable at the next height and a form-Z entry.
//!   The chain arm's preparation is kept for the producer.
//! - **A junction is valid over one parent's lanes.** A held batch is
//!   offered to a block only over exactly the lanes and leaves it was
//!   decided against; a batch decided against lanes the chain has moved past
//!   is never offered again, and leaves the queue when a higher fee needs
//!   its place.
//! - **Bounded work.** Cheap checks (v1.6 active, class, count) come before
//!   any fetch, payment verification or decision; at most two decisions run
//!   at once; a refused batch is remembered (bounded) and never fetched or
//!   decided again.
//! - **Not kept across a restart.** The queue is in memory: an aggregator
//!   submits again (its batch is decided against the tip anyway).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::client_objects::{
    check_batch_mask_carriable, check_batch_mask_form_z, pays_batch_submission, ClientBatchRules,
    ClientBatchSubmission, ClientObject, ClientObjectRules, ClientRegistryState,
    CLIENT_BATCH_CLASS, CLIENT_REGISTRY_CAPACITY,
};
use jetsam_ivc_core::field_r1cs::FieldR1cs;
use jetsam_p2p::client_object_protocol::{ClientBatchAnnouncement, ClientBatchBundle, ClientBatchId};
use jetsam_recursive::acceptance::history_step::HistoryStepRuntime;
use jetsam_recursive::{HistoryStepClientMatrices, HistoryStepClientMatrixSet, ParentLanes, PreparedBatch};
use jetsam_tx::PagedSpendIntent;

/// Batches held for the miner at once.
const BATCH_QUEUE_CAPACITY: usize = 8;

/// One batch in the queue: what it was decided against, what pays it, the
/// registry entries it uses, and what the producer takes (`payload`).
#[derive(Clone, Debug)]
pub struct QueuedBatch<P> {
    pub submission: ClientBatchSubmission,
    /// The transaction that pays it and carries its marker.
    pub payment: PagedSpendIntent,
    /// The parent lanes it was decided against: a block carries it only over
    /// exactly these.
    pub incoming: ParentLanes,
    /// The registry leaves it was decided against.
    pub leaves: Vec<[u8; 32]>,
    /// The registry entries it uses (rechecked carriable at each height).
    pub mask: Vec<bool>,
    pub payload: P,
}

/// Why the queue refused a batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchQueueError {
    MalformedPayment,
    NotThePayment,
    FeeTooLow { minimum: u64, paid: u64 },
    Duplicate,
    Full,
}

impl std::fmt::Display for BatchQueueError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[derive(Clone, Debug)]
struct QueuedEntry<P> {
    batch: QueuedBatch<P>,
    fee: u64,
    live_inputs: u16,
    live_outputs: u16,
    anchor: [u8; 32],
    spent: std::collections::BTreeSet<u32>,
    arrival: u64,
}

impl<P> QueuedEntry<P> {
    /// The worse of two: lower fee, then later arrival.
    fn rank(&self) -> (u64, std::cmp::Reverse<u64>) {
        (self.fee, std::cmp::Reverse(self.arrival))
    }
}

/// The batches held for the miner, bounded, by fee: the rules of the v1.5
/// client queue (`ClientSubmissionQueue`) plus the
/// lanes and leaves a junction was decided against.
#[derive(Debug)]
pub struct BatchQueue<P> {
    capacity: usize,
    next_arrival: u64,
    entries: Vec<QueuedEntry<P>>,
}

impl<P> BatchQueue<P> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            next_arrival: 0,
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn contains(&self, submission: &ClientBatchSubmission) -> bool {
        self.entries.iter().any(|entry| entry.batch.submission == *submission)
    }

    /// Hold `batch`: its payment opens exactly its submission and pays at
    /// least the submission fee. The same submission decided again (on
    /// other lanes or leaves) replaces the held one; decided on the same
    /// ones, it is a duplicate. When full, the lowest-paying held batch
    /// makes room for a strictly better one.
    pub fn insert(&mut self, batch: QueuedBatch<P>, rules: &ClientObjectRules) -> Result<(), BatchQueueError> {
        let spend = jetsam_tx::validate_paged_spend(&batch.payment.pages)
            .map_err(|_| BatchQueueError::MalformedPayment)?;
        if !pays_batch_submission(&batch.payment.pages, &batch.submission) {
            return Err(BatchQueueError::NotThePayment);
        }
        if spend.fee < rules.submission_fee_micro {
            return Err(BatchQueueError::FeeTooLow {
                minimum: rules.submission_fee_micro,
                paid: spend.fee,
            });
        }
        if let Some(index) = self.entries.iter().position(|entry| entry.batch.submission == batch.submission) {
            let held = &self.entries[index].batch;
            if held.incoming == batch.incoming && held.leaves == batch.leaves {
                return Err(BatchQueueError::Duplicate);
            }
            self.entries.swap_remove(index);
        }
        let spent = batch
            .payment
            .pages
            .iter()
            .flat_map(|page| page.body.live_inputs())
            .map(|(_, input)| input.slot_index)
            .collect();
        let entry = QueuedEntry {
            batch,
            fee: spend.fee,
            live_inputs: spend.live_inputs,
            live_outputs: spend.live_outputs,
            anchor: spend.epoch_anchor,
            spent,
            arrival: self.next_arrival,
        };
        if self.entries.len() >= self.capacity {
            let (index, worst) = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, entry)| entry.rank())
                .map(|(index, entry)| (index, entry.rank()))
                .expect("a full queue is not empty");
            if entry.rank() <= worst {
                return Err(BatchQueueError::Full);
            }
            self.entries.swap_remove(index);
        }
        self.next_arrival += 1;
        self.entries.push(entry);
        Ok(())
    }

    /// The batch the block after `parent` should carry: decided against
    /// exactly `lanes` and `leaves`, its payment minable (`anchor_ok`) and
    /// paying at least the required fee plus the submission fee, every entry
    /// it uses carriable at the block's height on `registry`; the highest
    /// fee, then the earliest received. Every filter before the maximum.
    pub fn best(
        &self,
        parent: &BlockHeader,
        lanes: &ParentLanes,
        leaves: &[[u8; 32]],
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    ) -> Option<&QueuedBatch<P>> {
        let height = parent.height.saturating_add(1);
        self.entries
            .iter()
            .filter(|entry| entry.batch.incoming == *lanes && entry.batch.leaves == leaves)
            .filter(|entry| anchor_ok(&entry.anchor))
            .filter(|entry| {
                let required = jetsam_chain::consensus::fees::fee_breakdown(
                    u64::from(entry.live_inputs),
                    u64::from(entry.live_outputs),
                    parent.active_slot_count,
                    parent.log_slots,
                )
                .required_total
                .saturating_add(rules.submission_fee_micro);
                entry.fee >= required
            })
            .filter(|entry| {
                check_batch_mask_carriable(&entry.batch.mask, leaves, registry, height, rules).is_ok()
            })
            .max_by_key(|entry| entry.rank())
            .map(|entry| &entry.batch)
    }

    /// Forget what a committed block settles: the batch it carried; every
    /// batch whose payment spends an input it spent; and, when it carried a
    /// batch or registered a client, every batch decided against the lanes
    /// or leaves it moved past (a block that does neither carries its
    /// parent's lanes and leaves).
    pub fn on_block_committed(&mut self, block: &jetsam_chain::Block) {
        let mut moved = false;
        for object in &block.client_objects {
            match object {
                ClientObject::BatchSubmission(submission) => {
                    self.forget(submission);
                    moved = true;
                }
                ClientObject::Registration(_) => moved = true,
                ClientObject::Submission(_) => {}
            }
        }
        if moved {
            self.entries.clear();
            return;
        }
        let spent: std::collections::BTreeSet<u32> = block
            .transactions
            .iter()
            .flat_map(|tx| tx.body.live_inputs())
            .map(|(_, input)| input.slot_index)
            .collect();
        self.entries.retain(|entry| entry.spent.is_disjoint(&spent));
    }

    /// Forget batches whose payment can no longer be mined.
    pub fn retain_minable(&mut self, anchor_ok: impl Fn(&[u8; 32]) -> bool) {
        self.entries.retain(|entry| anchor_ok(&entry.anchor));
    }

    /// Forget `submission`'s batch.
    pub fn forget(&mut self, submission: &ClientBatchSubmission) {
        self.entries.retain(|entry| entry.batch.submission != *submission);
    }

    /// The held batches decided over exactly `lanes` and `leaves`: those the
    /// block after a tip publishing them can carry.
    pub fn over<'a>(
        &'a self,
        lanes: &'a ParentLanes,
        leaves: &'a [[u8; 32]],
    ) -> impl Iterator<Item = &'a QueuedBatch<P>> + 'a {
        self.entries
            .iter()
            .map(|entry| &entry.batch)
            .filter(move |batch| batch.incoming == *lanes && batch.leaves == leaves)
    }

    pub fn iter(&self) -> impl Iterator<Item = &QueuedBatch<P>> {
        self.entries.iter().map(|entry| &entry.batch)
    }
}

/// What a held batch keeps besides its queue key.
#[derive(Clone, Debug)]
pub struct HeldBatch {
    pub id: ClientBatchId,
    pub bytes: Arc<[u8]>,
    pub payment: PagedSpendIntent,
    pub prepared: PreparedBatch,
}

/// The tip a batch is received against: its height, its terminal (none at
/// genesis) and the registry its branch ends with.
pub struct BatchTip<'a> {
    pub height: u64,
    pub terminal: Option<&'a [u8]>,
    pub registry: &'a ClientRegistryState,
}

/// Batch receptions decided at once: each is seconds
/// of native decision on the inbound verifier pool.
const MAX_CONCURRENT_RECEPTIONS: usize = 2;
/// Batch ids refused and not fetched or decided again (bounded).
const REFUSED_MEMORY: usize = 256;
/// How long a held batch goes unannounced before it is announced again, so
/// that a peer that connected after its announcement, or every peer when
/// its node had none at the time, still learns of it. Above gossipsub's
/// 60-second duplicate window (a gossip id is the bytes).
pub const BATCH_REANNOUNCE_INTERVAL: Duration = Duration::from_secs(120);

/// The node's batches (see the module documentation).
pub struct ClientBatches {
    runtime: Arc<HistoryStepRuntime>,
    matrices: Arc<HistoryStepClientMatrixSet>,
    rules: ClientObjectRules,
    batch_rules: ClientBatchRules,
    queue: Mutex<BatchQueue<HeldBatch>>,
    receptions: Receptions,
    reannouncements: Reannouncements,
}

/// When this node last announced each held batch. Bounded by the held
/// batches: an id no longer held is forgotten.
#[derive(Default)]
struct Reannouncements {
    last: Mutex<std::collections::HashMap<ClientBatchId, Instant>>,
}

impl Reannouncements {
    /// `id` was announced at `now`.
    fn announced(&self, id: ClientBatchId, now: Instant) {
        self.last.lock().expect("reannouncements lock").insert(id, now);
    }

    /// Among `held` (id, fee), the announcements due at `now`: those never
    /// announced here, or not for [`BATCH_REANNOUNCE_INTERVAL`]; each is
    /// marked announced at `now`. Ids no longer held are forgotten.
    fn due(&self, held: &[(ClientBatchId, u64)], now: Instant) -> Vec<ClientBatchAnnouncement> {
        let mut last = self.last.lock().expect("reannouncements lock");
        last.retain(|id, _| held.iter().any(|(held_id, _)| held_id == id));
        let mut due = Vec::new();
        for &(id, fee) in held {
            let is_due = last
                .get(&id)
                .is_none_or(|at| now.saturating_duration_since(*at) >= BATCH_REANNOUNCE_INTERVAL);
            if is_due {
                last.insert(id, now);
                due.push(ClientBatchAnnouncement { id, fee });
            }
        }
        due
    }
}

/// The registry leaves a block publishes over `registry`, zero-padded.
pub fn registry_leaves(registry: &ClientRegistryState) -> Vec<[u8; 32]> {
    let mut leaves = registry.digests();
    leaves.resize(CLIENT_REGISTRY_CAPACITY.max(leaves.len()), [0u8; 32]);
    leaves
}

/// Releases one reception slot when dropped.
struct ReceptionSlot<'a>(&'a std::sync::atomic::AtomicUsize);

impl Drop for ReceptionSlot<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

/// What bounds the work receptions cost: at most
/// [`MAX_CONCURRENT_RECEPTIONS`] decisions at once, and the batches refused
/// here (bounded by [`REFUSED_MEMORY`]), never decided again.
#[derive(Default)]
struct Receptions {
    refused: Mutex<std::collections::VecDeque<ClientBatchId>>,
    /// Batches refused for what the chain may still change (`retry:`), with
    /// the tip they were refused at (bounded by [`REFUSED_MEMORY`]).
    retried: Mutex<std::collections::VecDeque<(ClientBatchId, u64)>>,
    receiving: std::sync::atomic::AtomicUsize,
}

impl Receptions {
    /// A reception slot, or `busy` while [`MAX_CONCURRENT_RECEPTIONS`] are taken.
    fn admit(&self) -> Result<ReceptionSlot<'_>, String> {
        use std::sync::atomic::Ordering;
        if self.receiving.fetch_add(1, Ordering::AcqRel) >= MAX_CONCURRENT_RECEPTIONS {
            self.receiving.fetch_sub(1, Ordering::AcqRel);
            return Err("busy: other batches are being decided".into());
        }
        Ok(ReceptionSlot(&self.receiving))
    }

    fn was_refused(&self, id: &ClientBatchId) -> bool {
        self.refused.lock().expect("refused batches lock").contains(id)
    }

    /// Remember `id` when its decision refused the batch itself.
    ///
    /// Remembered only when the batch itself is at fault (its proof, its
    /// statement, its entries' form, its fee); never for what the chain
    /// may still change (an entry not carriable yet, a full queue, the
    /// tip's lanes unreadable) — those are prefixed `retry:`.
    fn settle<T>(&self, id: ClientBatchId, outcome: &Result<T, String>) {
        if let Err(error) = outcome {
            if !error.starts_with("held:") && !error.starts_with("retry:") {
                self.remember_refused(id);
            }
        }
    }

    /// Remember `id` refused at the tip `tip_height` for what the chain may
    /// still change (`retry:`): it is not fetched again before the tip
    /// moves, however often it is announced.
    fn note_retry<T>(&self, id: ClientBatchId, outcome: &Result<T, String>, tip_height: u64) {
        if let Err(error) = outcome {
            if error.starts_with("retry:") {
                let mut retried = self.retried.lock().expect("retried batches lock");
                retried.retain(|(known, _)| *known != id);
                if retried.len() >= REFUSED_MEMORY {
                    retried.pop_front();
                }
                retried.push_back((id, tip_height));
            }
        }
    }

    /// Whether `id` was refused `retry:` at the tip `tip_height`.
    fn waits_for_next_tip(&self, id: &ClientBatchId, tip_height: u64) -> bool {
        self.retried
            .lock()
            .expect("retried batches lock")
            .iter()
            .any(|(known, height)| known == id && *height == tip_height)
    }

    fn remember_refused(&self, id: ClientBatchId) {
        let mut refused = self.refused.lock().expect("refused batches lock");
        if !refused.contains(&id) {
            if refused.len() >= REFUSED_MEMORY {
                refused.pop_front();
            }
            refused.push_back(id);
        }
    }
}

impl ClientBatches {
    /// `runtime` is the v1.6 relation, `matrices` the form-Z matrix set.
    pub fn new(
        runtime: Arc<HistoryStepRuntime>,
        matrices: Arc<HistoryStepClientMatrixSet>,
        rules: ClientObjectRules,
        batch_rules: ClientBatchRules,
    ) -> Self {
        Self {
            runtime,
            matrices,
            rules,
            batch_rules,
            queue: Mutex::new(BatchQueue::new(BATCH_QUEUE_CAPACITY)),
            receptions: Receptions::default(),
            reannouncements: Reannouncements::default(),
        }
    }

    /// The cheap checks of a batch submission, before anything is fetched,
    /// verified or decided: a batch is carried at the block after the tip,
    /// its class is the chain's, its count within the bound.
    pub fn precheck(&self, submission: &ClientBatchSubmission, tip_height: u64) -> Result<(), String> {
        let child = tip_height.saturating_add(1);
        if !self.batch_rules.active_at(child) {
            return Err(format!("no batch is carried at height {child}: below the v1.6 height"));
        }
        if CLIENT_BATCH_CLASS != Some(submission.batch_class) {
            return Err("the batch class is not the chain's (CLIENT_BATCH_CLASS)".into());
        }
        if submission.count == 0 || submission.count > self.batch_rules.max_clients {
            return Err(format!(
                "a batch covers 1 to {} clients, not {}",
                self.batch_rules.max_clients, submission.count
            ));
        }
        Ok(())
    }

    /// Whether `id` was refused here (it is neither fetched nor decided again).
    pub fn was_refused(&self, id: &ClientBatchId) -> bool {
        self.receptions.was_refused(id)
    }

    /// Whether `id` was refused `retry:` against the tip at `tip_height`: it
    /// is not fetched again before the tip moves.
    pub fn waits_for_next_tip(&self, id: &ClientBatchId, tip_height: u64) -> bool {
        self.receptions.waits_for_next_tip(id, tip_height)
    }

    /// The lanes the block after the tip at `tip_height` starts from: dead
    /// at the first v1.6 block (its relation's base), else the lanes the
    /// tip's `terminal` publishes. Refused below the v1.6 height.
    pub fn parent_lanes(&self, tip_height: u64, terminal: Option<&[u8]>) -> Result<ParentLanes, String> {
        let child = tip_height.saturating_add(1);
        if !self.batch_rules.active_at(child) {
            return Err(format!("no batch is carried at height {child}: below the v1.6 height"));
        }
        if self.runtime.bank().recursion_root_height() == tip_height {
            return Ok(ParentLanes::empty(CLIENT_REGISTRY_CAPACITY));
        }
        let terminal = terminal.ok_or("the tip's terminal is missing")?;
        jetsam_recursive::history_step_terminal_parent_lanes(&self.runtime, terminal)
    }

    /// The form-Z matrices this node holds for `leaves`.
    fn held_matrices(&self, leaves: &[[u8; 32]]) -> BTreeMap<[u8; 32], Arc<FieldR1cs>> {
        leaves
            .iter()
            .filter(|leaf| **leaf != [0u8; 32])
            .filter_map(|leaf| self.matrices.client_matrix(leaf).map(|matrix| (*leaf, matrix)))
            .collect()
    }

    /// Receive a batch bundle against `tip` (CPU-heavy: a native decision,
    /// seconds; call on a blocking worker): decoded (exactly the bytes of
    /// `expected` when it was fetched by its id), prechecked, decided,
    /// checked and held. At most [`MAX_CONCURRENT_RECEPTIONS`] at once; a
    /// refused batch is remembered and never decided again. Returns its
    /// announcement.
    pub fn receive(
        &self,
        bytes: &[u8],
        tip: &BatchTip<'_>,
        expected: Option<&ClientBatchId>,
    ) -> Result<ClientBatchAnnouncement, String> {
        let _slot = self.receptions.admit()?;
        let bundle = match expected {
            Some(id) => ClientBatchBundle::decode_for(id, bytes),
            None => ClientBatchBundle::decode(bytes),
        }
        .map_err(|error| format!("batch bundle: {error}"))?;
        let id = ClientBatchId::of_bytes(bundle.submission, bytes).ok_or("batch bundle too long")?;
        if self.was_refused(&id) {
            return Err("this batch was refused here already".into());
        }
        self.precheck(&bundle.submission, tip.height)?;
        let outcome = self.decide_and_hold(id, bytes, bundle, tip);
        self.receptions.settle(id, &outcome);
        self.receptions.note_retry(id, &outcome, tip.height);
        outcome
    }

    fn decide_and_hold(
        &self,
        id: ClientBatchId,
        bytes: &[u8],
        bundle: ClientBatchBundle,
        tip: &BatchTip<'_>,
    ) -> Result<ClientBatchAnnouncement, String> {
        let child = tip.height.saturating_add(1);
        let lanes = self
            .parent_lanes(tip.height, tip.terminal)
            .map_err(|error| format!("retry: {error}"))?;
        let leaves = registry_leaves(tip.registry);
        if self
            .queue
            .lock()
            .expect("batch queue lock")
            .iter()
            .any(|held| held.payload.id == id && held.incoming == lanes && held.leaves == leaves)
        {
            return Err("held: this batch is held already".into());
        }
        let class = jetsam_recursive::AggregationClass::chain()?;
        let decided = jetsam_recursive::decide_client_batch_bytes(
            class,
            &bundle.proof_bytes,
            &leaves,
            &lanes,
            &self.held_matrices(&leaves),
        )?;
        if decided.statement.statements_root != bundle.submission.statements_root
            || decided.statement.count != u64::from(bundle.submission.count)
        {
            return Err("the batch proof is not the batch its submission names".into());
        }
        check_batch_mask_form_z(&decided.statement.mask, &leaves, &self.batch_rules)
            .map_err(|error| format!("batch entries: {error}"))?;
        check_batch_mask_carriable(&decided.statement.mask, &leaves, tip.registry, child, &self.rules)
            .map_err(|error| format!("retry: batch entries not carriable at height {child}: {error}"))?;
        let held = QueuedBatch {
            submission: bundle.submission,
            payment: bundle.payment.clone(),
            incoming: lanes,
            leaves,
            mask: decided.statement.mask.clone(),
            payload: HeldBatch {
                id,
                bytes: Arc::from(bytes),
                payment: bundle.payment,
                prepared: decided.prepared,
            },
        };
        let fee = jetsam_tx::validate_paged_spend(&held.payment.pages)
            .map(|spend| spend.fee)
            .unwrap_or_default();
        self.queue
            .lock()
            .expect("batch queue lock")
            .insert(held, &self.rules)
            .map_err(|error| match error {
                BatchQueueError::Duplicate => "held: this batch is held already".to_owned(),
                BatchQueueError::Full => format!("retry: batch queue: {error}"),
                error => format!("batch queue: {error}"),
            })?;
        tracing::info!(
            height = child,
            statements_root = %hex::encode(bundle.submission.statements_root),
            count = bundle.submission.count,
            fee,
            "client batch decided and held for the miner"
        );
        // The caller announces it now.
        self.reannouncements.announced(id, Instant::now());
        Ok(ClientBatchAnnouncement { id, fee })
    }

    /// The batch the block after `parent` should carry, on `registry` (the
    /// registry the parent's branch ends with).
    pub fn best(
        &self,
        parent: &BlockHeader,
        parent_terminal: Option<&[u8]>,
        registry: &ClientRegistryState,
        anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    ) -> Option<jetsam_miner::client_slot::MinerBatch> {
        let lanes = match self.parent_lanes(parent.height, parent_terminal) {
            Ok(lanes) => lanes,
            Err(error) => {
                tracing::debug!(%error, "no batch lanes for this parent");
                return None;
            }
        };
        let leaves = registry_leaves(registry);
        let queue = self.queue.lock().expect("batch queue lock");
        let batch = queue.best(parent, &lanes, &leaves, registry, &self.rules, anchor_ok)?;
        Some(jetsam_miner::client_slot::MinerBatch {
            submission: batch.submission,
            payment: batch.payload.payment.clone(),
            fee: jetsam_tx::validate_paged_spend(&batch.payment.pages)
                .map(|spend| spend.fee)
                .unwrap_or_default(),
            prepared: batch.payload.prepared.clone(),
        })
    }

    /// The exact bytes of a held batch.
    pub fn bytes(&self, id: &ClientBatchId) -> Option<Vec<u8>> {
        self.queue
            .lock()
            .expect("batch queue lock")
            .iter()
            .find(|batch| batch.payload.id == *id)
            .map(|batch| batch.payload.bytes.to_vec())
    }

    /// Whether the exact batch `id` is held.
    pub fn holds(&self, id: &ClientBatchId) -> bool {
        self.queue
            .lock()
            .expect("batch queue lock")
            .iter()
            .any(|batch| batch.payload.id == *id)
    }

    /// The held batches due to be announced again at `now`, marked
    /// announced: a held batch the block after the tip can carry (decided
    /// over its `lanes` and `leaves`) goes at most
    /// [`BATCH_REANNOUNCE_INTERVAL`] without an announcement; one decided
    /// over lanes the chain has moved past is not announced again.
    pub fn announcements_due(
        &self,
        now: Instant,
        lanes: &ParentLanes,
        leaves: &[[u8; 32]],
    ) -> Vec<ClientBatchAnnouncement> {
        let held: Vec<(ClientBatchId, u64)> = self
            .queue
            .lock()
            .expect("batch queue lock")
            .over(lanes, leaves)
            .map(|batch| {
                let fee = jetsam_tx::validate_paged_spend(&batch.payment.pages)
                    .map(|spend| spend.fee)
                    .unwrap_or_default();
                (batch.payload.id, fee)
            })
            .collect();
        self.reannouncements.due(&held, now)
    }

    /// Held batches.
    pub fn len(&self) -> usize {
        self.queue.lock().expect("batch queue lock").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// A template refused the block carrying `submission`'s batch: it is
    /// not offered again.
    pub fn forget(&self, submission: &ClientBatchSubmission) {
        self.queue.lock().expect("batch queue lock").forget(submission);
    }

    /// A block was committed: what it settles leaves the queue
    /// ([`BatchQueue::on_block_committed`]).
    pub fn on_block_committed(&self, block: &jetsam_chain::Block) {
        self.queue.lock().expect("batch queue lock").on_block_committed(block);
    }
}

#[cfg(test)]
mod tests;
