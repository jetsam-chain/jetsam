// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! `AsyncMempool` — async wrapper around the synchronous `jetsam_chain::Mempool`.
//!
//! ## Architecture
//!
//! ```text
//!  submit(PagedSpendIntent)
//!    │
//!    ├─ Stateless check (no lock): canonical body logic + derived txid
//!    │
//!    ├─ Pre-proof filter (lock, brief): all cheap checks on current view
//!    │   fee floor → consensus → epoch_anchor → slot conflicts → slot state
//!    │   Extracts log_slots: u32 only — NO ChainView clone.
//!    │   DoS guard: invalid txs rejected here, never reach proof verification.
//!    │
//!    ├─ selected-ZK authorization verification (no lock), semaphore-bounded
//!    │
//!    └─ Final admission (lock): re-run all checks against current state (TOCTOU guard)
//!                        anchor_height derived here → pool.admit
//!
//!  on_new_block() ──► [remove confirmed] ──► [evict expired] ──► [update chain view]
//!
//!  select_for_block() ──► fee-sorted list of MempoolEntry (verified txs only)
//! ```
//!
//! ## Pre-proving cache
//!
//! When a wallet submits a `PagedSpendIntent`, it includes a `WalletAuthorizationBundle`
//! (one versioned witness-hiding proof). The pool retains one immutable intent
//! allocation and borrows the bundle suffix from it during block assembly.
//! The block assembler uses cached bundles so that `prove_block` only
//! needs to run the unified block-level SpineGKR + single FRI — the
//! per-tx wallet work is already done.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::{broadcast, Mutex, Semaphore};

use jetsam_chain::consensus::params::BLOCK_MAX_USER_PAGES;
use jetsam_chain::consensus::wire_limits::{
    MAX_AUTHORIZATION_BYTES, MAX_MEMPOOL_TXS, MAX_TX_INTENT_BYTES_GLOBAL,
};
use jetsam_chain::consensus::{fee_breakdown, tx_epoch_anchor_height_for_child};
use jetsam_chain::fri_state::SlotValue;
use jetsam_chain::Mempool;
use jetsam_poseidon2b::primitives::{Address, TxBodyHash};
use jetsam_tx::{
    validate_paged_spend, PagedSpendFacts, PagedSpendIntent, TxPage, PAGED_SPEND_INTENT_MARKER,
};

use crate::config::MempoolConfig;
use crate::error::SubmitError;
use crate::event::{EvictReason, MempoolEvent};
use crate::floor::{FeeFloor, RelayFloor};
use crate::view::ChainView;

/// One CPU-heavy authorization verification ready to run on a node-owned
/// executor. The mempool owns protocol validation; the embedding node owns
/// process-wide CPU admission.
pub type AuthorizationVerificationTask = Box<dyn FnOnce() -> Result<(), String> + Send + 'static>;

/// Executor hook for authorization verification. The default runs the task on
/// the surrounding `spawn_blocking` thread. Production nodes replace it with
/// their common proof Rayon pool so mempool traffic cannot activate an
/// independent global pool beside Block/Link/Verify.
pub type AuthorizationVerificationExecutor =
    Arc<dyn Fn(AuthorizationVerificationTask) -> Result<(), String> + Send + Sync + 'static>;

// ---------------------------------------------------------------------------
// Internal state (held under Mutex)
// ---------------------------------------------------------------------------

pub(crate) struct MempoolState {
    /// Synchronous core pool (conflict tracking, fee ordering).
    pub pool: Mempool,
    /// Chain view snapshot — updated on every new block.
    pub view: ChainView,
    /// Dynamic fee floor.
    pub floor: FeeFloor,
    /// Input slot indices currently held by admitted txs. O(1) conflict check.
    pub admitted_input_slots: HashSet<u32>,
    /// Output slot indices currently held by admitted txs. O(1) conflict check.
    pub admitted_output_slots: HashSet<u32>,
    /// The v1.5 client-object rules plain transactions are judged by
    /// ([`MempoolConfig::client_object_rules`]).
    pub client_rules: jetsam_chain::consensus::client_objects::ClientObjectRules,
    /// The v1.7 vault rules plain transactions are judged by
    /// ([`MempoolConfig::vault_rules`]).
    pub vault_rules: jetsam_chain::consensus::vault::VaultRules,
    /// v1.7: destination objects held with their transactions, never in the
    /// fee pool ([`crate::destination`]).
    pub destination: crate::destination::DestinationPool,
    /// v1.7: guardian objects held with their transactions, never in the fee
    /// pool ([`crate::guardian`]).
    pub guardian: crate::guardian::GuardianPool,
    /// The v1.7 guardian rules ([`MempoolConfig::guardian_rules`]).
    pub guardian_rules: jetsam_chain::consensus::guardian::GuardianRules,
}

/// Compact immutable RPC/diagnostic projection of one mempool entry.
///
/// This type deliberately contains no intent or authorization byte vector, so
/// inspecting a full pool cannot duplicate its retained proof payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MempoolEntryMetadata {
    pub tx_hash: TxBodyHash,
    pub fee_micro_jtm: u64,
    pub fee_rate: u64,
    pub n_inputs: u16,
    pub n_outputs: u16,
    pub page_count: u16,
    pub admitted_height: u64,
    pub has_authorization: bool,
}

/// One lock-consistent compact view of fee floor and all entry metadata.
#[derive(Debug, PartialEq, Eq)]
pub struct MempoolMetadataSnapshot {
    pub fee_floor: RelayFloor,
    pub entries: Vec<MempoolEntryMetadata>,
}

/// Constant-size lock-consistent mempool pressure snapshot for status UIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MempoolUsageSnapshot {
    pub size: usize,
    pub capacity: usize,
    pub intent_bytes: usize,
    pub max_intent_bytes: usize,
    pub fee_floor: RelayFloor,
}

/// Minimal owned block-template selection.
///
/// Raw `PagedSpendIntent` bytes are a networking cache and are never cloned into the
/// miner. Only the semantic body and the cached authorization bundle cross the
/// mempool lock.
#[derive(Debug)]
pub struct SelectedMempoolEntry {
    pub pages: Vec<TxPage>,
    pub logical_txid: TxBodyHash,
    pub cached_authorization: Option<Vec<u8>>,
}

fn entry_metadata(
    hash: TxBodyHash,
    entry: &jetsam_chain::mempool::MempoolEntry,
) -> MempoolEntryMetadata {
    MempoolEntryMetadata {
        tx_hash: hash,
        fee_micro_jtm: entry.spend.fee,
        fee_rate: entry.fee_rate,
        n_inputs: entry.spend.live_inputs,
        n_outputs: entry.spend.live_outputs,
        page_count: entry.pages.len() as u16,
        admitted_height: entry.admitted_height,
        has_authorization: entry.cached_authorization().is_some(),
    }
}

// ---------------------------------------------------------------------------
// AsyncMempool
// ---------------------------------------------------------------------------

/// Async, thread-safe mempool for the Jetsam full node.
///
/// Clone is O(1) — the inner state is reference-counted.
#[derive(Clone)]
pub struct AsyncMempool {
    state: Arc<Mutex<MempoolState>>,
    events: broadcast::Sender<MempoolEvent>,
    config: Arc<MempoolConfig>,
    /// Semaphore limiting concurrent authorization verification tasks.
    /// Bounds CPU usage: at most `config.auth_verify_workers` proofs in flight.
    /// Set to 0 in config → semaphore with MAX permits (no limit).
    auth_verify_semaphore: Arc<Semaphore>,
    auth_verify_executor: AuthorizationVerificationExecutor,
    /// v1.7: guardian BLS verifications run so far (A1 m13: each one comes
    /// after every cheap refusal and within the rate limit).
    guardian_signature_checks: Arc<std::sync::atomic::AtomicU64>,
}

/// Largest number of events one `on_new_block` can emit before any subscriber
/// gets a chance to run: every transaction in the block confirmed, and — at an
/// epoch boundary — every transaction left in the pool evicted, all inside one
/// lock.
///
/// The channel used to hold exactly `MAX_MEMPOOL_TXS`, which is the eviction
/// half alone, so the confirmations on top of it had nowhere to go. A dropped
/// event is not merely a missing log line: this channel is how the node's
/// wallet learns that one of its own sends will never confirm, and a lagged
/// receiver silently leaves that send's coins reserved for the life of the
/// process — the exact bug `forget_evicted_send` exists to prevent, reappearing
/// only under load, which is when nobody is watching.
const EVENT_CHANNEL_CAPACITY: usize = MAX_MEMPOOL_TXS + BLOCK_MAX_USER_PAGES + 1;

impl AsyncMempool {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new empty mempool with the given initial chain view.
    pub fn new(view: ChainView, config: MempoolConfig) -> Self {
        let (events, _) = broadcast::channel(EVENT_CHANNEL_CAPACITY);
        let floor = FeeFloor::new(config.fee_floor_window);
        let state = MempoolState {
            pool: Mempool::new(config.capacity),
            view,
            floor,
            admitted_input_slots: HashSet::new(),
            admitted_output_slots: HashSet::new(),
            client_rules: config.client_object_rules,
            vault_rules: config.vault_rules,
            destination: Default::default(),
            guardian: Default::default(),
            guardian_rules: config.guardian_rules,
        };
        let max_permits = if config.auth_verify_workers == 0 {
            // 0 = unlimited concurrency; verification is still required
            usize::MAX / 2 // Semaphore::MAX_PERMITS
        } else {
            config.auth_verify_workers
        };
        let auth_verify_semaphore = Arc::new(Semaphore::new(max_permits));
        Self {
            state: Arc::new(Mutex::new(state)),
            events,
            config: Arc::new(config),
            auth_verify_semaphore,
            auth_verify_executor: Arc::new(|task| task()),
            guardian_signature_checks: Arc::default(),
        }
    }

    /// Route authorization proof work through a process-owned CPU executor.
    /// Configure this before cloning the mempool into RPC/P2P tasks.
    pub fn with_authorization_verification_executor(
        mut self,
        executor: AuthorizationVerificationExecutor,
    ) -> Self {
        self.auth_verify_executor = executor;
        self
    }

    /// Subscribe to mempool events (P2P, RPC WebSocket subscriptions, miner wakeup).
    pub fn subscribe(&self) -> broadcast::Receiver<MempoolEvent> {
        self.events.subscribe()
    }

    // -----------------------------------------------------------------------
    // Tx submission
    // -----------------------------------------------------------------------

    /// Submit one complete `PagedSpendIntent` for admission.
    ///
    /// Runs the full native admission pipeline:
    /// 1. Fee ≥ dynamic floor
    /// 2. Basic consensus checks (fee overflow, body hash, anchor)
    /// 3. epoch_anchor equals the one canonical next-block epoch anchor
    /// 4. No slot conflict with admitted mempool txs
    /// 5. Input slots live in state, output slots empty
    ///
    /// Selected-ZK authorization verification is performed synchronously (in a
    /// `spawn_blocking` task) BEFORE the pool mutex is acquired, so invalid
    /// proofs are rejected at the mempool boundary without holding the lock.
    ///
    /// Returns the `TxBodyHash` on success.
    pub async fn submit(
        &self,
        intent: PagedSpendIntent,
        intent_bytes: Vec<u8>,
    ) -> Result<TxBodyHash, SubmitError> {
        if intent_bytes.len() > MAX_TX_INTENT_BYTES_GLOBAL {
            return Err(SubmitError::IntentTooLarge {
                actual: intent_bytes.len(),
                max: MAX_TX_INTENT_BYTES_GLOBAL,
            });
        }
        if !canonical_intent_bytes_match(&intent, &intent_bytes) {
            return Err(SubmitError::MalformedIntent(
                "decoded intent does not match retained canonical wire bytes".into(),
            ));
        }
        // ── Stateless sanity check (no lock, no IO) ────────────────────
        // Revalidate group semantics at this trust boundary even when the
        // caller already used the bounded decoder.
        let spend = validate_paged_spend(&intent.pages)
            .map_err(|e| SubmitError::MalformedIntent(format!("PagedSpend: {e}")))?;
        let txid = spend.logical_txid;

        if intent.authorization_bytes.is_empty() {
            return Err(SubmitError::MissingProof);
        }
        if intent.authorization_bytes.len() > MAX_AUTHORIZATION_BYTES {
            return Err(SubmitError::ProofTooLarge {
                actual: intent.authorization_bytes.len(),
                max: MAX_AUTHORIZATION_BYTES,
            });
        }
        // ── Cheap pre-filter (lock held briefly) ─────────────────
        // Runs all cheap state checks before expensive Auth verification.
        {
            let st = self.state.lock().await;
            if st.pool.contains(&txid) {
                return Err(SubmitError::AlreadyAdmitted(txid));
            }
            let projected_bytes = st
                .pool
                .total_intent_bytes()
                .saturating_add(intent_bytes.len());
            if projected_bytes > self.config.max_total_intent_bytes {
                return Err(SubmitError::BytesFull {
                    actual: projected_bytes,
                    max: self.config.max_total_intent_bytes,
                });
            }
            let _ = run_admission_checks(&intent.pages, &spend, &st)?;
        }

        // ── Authorization verification (CPU-heavy, outside lock, semaphore-bounded) ─
        // Runs only when the pre-filter passed — invalid fee/anchor/slot txs are
        // already gone.  Semaphore caps concurrent CPU threads.
        {
            let proof_bytes = intent.authorization_bytes.clone();
            let pages = intent.pages.clone();
            let executor = Arc::clone(&self.auth_verify_executor);

            let _permit =
                self.auth_verify_semaphore.acquire().await.map_err(|_| {
                    SubmitError::Internal("proof-verification semaphore closed".into())
                })?;

            tokio::task::spawn_blocking(move || {
                executor(Box::new(move || {
                    verify_intent_authorization(&pages, &proof_bytes)
                }))
            })
            .await
            .map_err(|e| SubmitError::Internal(format!("spawn_blocking: {e}")))?
            .map_err(SubmitError::InvalidProof)?;
        }

        // ── Final admission under lock ───────────────────────
        // Re-run all cheap checks against CURRENT state: the chain may have
        // advanced during the authorization verification window (new block → new
        // spent slots and changed fee floor). This is the
        // authoritative check; the pre-filter was the DoS guard.
        let mut st = self.state.lock().await;

        let hash = spend.logical_txid;

        if st.pool.contains(&hash) {
            return Err(SubmitError::AlreadyAdmitted(hash));
        }
        let projected_bytes = st
            .pool
            .total_intent_bytes()
            .saturating_add(intent_bytes.len());
        if projected_bytes > self.config.max_total_intent_bytes {
            return Err(SubmitError::BytesFull {
                actual: projected_bytes,
                max: self.config.max_total_intent_bytes,
            });
        }

        // Re-derive anchor_height from current state (needed by pool.admit).
        let _anchor_height = run_admission_checks(&intent.pages, &spend, &st)?;

        // --- Admit ---
        let fee = spend.fee;
        let tip_height = st.view.tip_height;
        match st.pool.admit(intent.pages, tip_height) {
            Ok(()) => {
                // Maintain persistent slot sets so future checks are O(1).
                let entry = st
                    .pool
                    .get(&hash)
                    .expect("newly admitted PagedSpend is indexed by logical txid");
                let input_slots: Vec<_> = entry
                    .pages
                    .iter()
                    .flat_map(|page| page.body.live_inputs())
                    .map(|(_, input)| input.slot_index)
                    .collect();
                let output_slots: Vec<_> = entry
                    .pages
                    .iter()
                    .flat_map(|page| page.body.live_outputs())
                    .map(|(_, output)| output.slot_index)
                    .collect();
                for slot in input_slots {
                    st.admitted_input_slots.insert(slot);
                }
                for slot in output_slots {
                    st.admitted_output_slots.insert(slot);
                }
            }
            Err(jetsam_chain::mempool::MempoolError::Full) => {
                return Err(SubmitError::Full {
                    capacity: self.config.capacity,
                });
            }
            Err(jetsam_chain::mempool::MempoolError::AlreadyAdmitted) => {
                return Err(SubmitError::AlreadyAdmitted(hash));
            }
            Err(e) => {
                return Err(SubmitError::Internal(format!("{e:?}")));
            }
        }

        // One immutable allocation backs durable mempool serving and every
        // broadcast subscriber. The miner's cached authorization is a borrowed
        // suffix of this allocation rather than a second retained proof copy.
        let intent_bytes: Arc<[u8]> = intent_bytes.into();
        st.pool.set_intent_bytes(&hash, Arc::clone(&intent_bytes));
        let _ = self.events.send(MempoolEvent::TxAdmitted {
            hash,
            fee,
            intent_bytes: Arc::clone(&intent_bytes),
        });
        let _ = self
            .events
            .send(MempoolEvent::TxAuthorizationVerified { hash });

        tracing::debug!(
            hash = ?hash,
            fee = fee,
            tip = st.view.tip_height,
            pool_size = st.pool.len(),
            "tx admitted to mempool"
        );

        Ok(hash)
    }

    /// Check a transaction that pays a v1.5 client object (M3.8). It travels
    /// with its object — inside a client proof bundle, or a registration
    /// handed to this node — and is never admitted to the pool, so it is
    /// checked here like an admission against the current state (fee, epoch
    /// anchor, live inputs, empty outputs, no spend from a locked address)
    /// but without the pool's own rules (its marker, conflicts with pool
    /// entries), then its authorization proof is verified, outside the lock.
    pub async fn check_client_payment(&self, intent: &PagedSpendIntent) -> Result<(), SubmitError> {
        let spend = validate_paged_spend(&intent.pages)
            .map_err(|e| SubmitError::MalformedIntent(format!("PagedSpend: {e}")))?;
        if intent.authorization_bytes.is_empty() {
            return Err(SubmitError::MissingProof);
        }
        if intent.authorization_bytes.len() > MAX_AUTHORIZATION_BYTES {
            return Err(SubmitError::ProofTooLarge {
                actual: intent.authorization_bytes.len(),
                max: MAX_AUTHORIZATION_BYTES,
            });
        }
        {
            let st = self.state.lock().await;
            run_admission_checks_with(&intent.pages, &spend, &st, Admission::ObjectPayment)?;
        }
        let proof_bytes = intent.authorization_bytes.clone();
        let pages = intent.pages.clone();
        let executor = Arc::clone(&self.auth_verify_executor);
        let _permit = self
            .auth_verify_semaphore
            .acquire()
            .await
            .map_err(|_| SubmitError::Internal("proof-verification semaphore closed".into()))?;
        tokio::task::spawn_blocking(move || {
            executor(Box::new(move || {
                verify_intent_authorization(&pages, &proof_bytes)
            }))
        })
        .await
        .map_err(|e| SubmitError::Internal(format!("spawn_blocking: {e}")))?
        .map_err(SubmitError::InvalidProof)
    }

    // -----------------------------------------------------------------------
    // v1.7 destination objects (T2.5)
    // -----------------------------------------------------------------------

    /// Admit one encoded destination intent ([`crate::destination`]): a
    /// committee update with its payment, or a release with its finality
    /// update, account proof and release group. Checked here: the bytes, the
    /// v1.7 clock of the next block, the transaction against the current
    /// state (fee, epoch anchor, live inputs, empty outputs), a committee
    /// payment's marker, a release's index unpaid on the branch and its group
    /// against the vault gate (`check_release_group`); then, outside the
    /// lock, the L1 rules of the block with the committee keys of the view
    /// (A2-M1, [`crate::destination::verify_destination_intent`]: BLS, the
    /// bridge account, V7), then the transaction's authorization. Held, never
    /// put in the fee pool; the miners are woken.
    pub async fn submit_destination(
        &self,
        bytes: &[u8],
    ) -> Result<crate::destination::DestinationAdmitted, SubmitError> {
        use crate::destination::{DestinationAdmitted, DestinationIntent, HoldError};
        // A release over Base is read only under rules that arm its height.
        let over_base_on_the_wire = self
            .state
            .lock()
            .await
            .view
            .destination_rules
            .base_return
            .activation_height
            .is_some();
        let intent = DestinationIntent::decode_with(bytes, over_base_on_the_wire).map_err(
            |error| SubmitError::MalformedIntent(format!("destination intent: {error}")),
        )?;
        let payment = intent.payment();
        let spend = validate_paged_spend(&payment.pages)
            .map_err(|e| SubmitError::MalformedIntent(format!("PagedSpend: {e}")))?;
        if payment.authorization_bytes.is_empty() {
            return Err(SubmitError::MissingProof);
        }
        if payment.authorization_bytes.len() > MAX_AUTHORIZATION_BYTES {
            return Err(SubmitError::ProofTooLarge {
                actual: payment.authorization_bytes.len(),
                max: MAX_AUTHORIZATION_BYTES,
            });
        }
        let judge = {
            let st = self.state.lock().await;
            check_destination_admission(&intent, &spend, &st)?
        };
        // v1.7 (A2-M1): the L1 rules of the block, before the authorization
        // proof and outside the lock.
        let (intent, judge, verified) = tokio::task::spawn_blocking(move || {
            let verified = judge.verify(&intent);
            (intent, judge, verified)
        })
        .await
        .map_err(|e| SubmitError::Internal(format!("spawn_blocking: {e}")))?;
        verified?;
        let payment = intent.payment();
        let proof_bytes = payment.authorization_bytes.clone();
        let pages = payment.pages.clone();
        let executor = Arc::clone(&self.auth_verify_executor);
        {
            let _permit =
                self.auth_verify_semaphore.acquire().await.map_err(|_| {
                    SubmitError::Internal("proof-verification semaphore closed".into())
                })?;
            tokio::task::spawn_blocking(move || {
                executor(Box::new(move || {
                    verify_intent_authorization(&pages, &proof_bytes)
                }))
            })
            .await
            .map_err(|e| SubmitError::Internal(format!("spawn_blocking: {e}")))?
            .map_err(SubmitError::InvalidProof)?;
        }
        let mut st = self.state.lock().await;
        let current = check_destination_admission(&intent, &spend, &st)?;
        // The view moved to other committees meanwhile: judged again by them.
        if !current.judges_like(&judge) {
            current.verify(&intent)?;
        }
        let admitted = match &intent {
            DestinationIntent::CommitteeUpdate(update) => DestinationAdmitted::CommitteeUpdate {
                period: update.next_period().unwrap_or_default(),
            },
            DestinationIntent::Release(release) if release.base.is_some() => {
                DestinationAdmitted::ReleaseOverBase {
                    n: release.release.n,
                }
            }
            DestinationIntent::Release(release) => DestinationAdmitted::Release {
                n: release.release.n,
            },
        };
        st.destination.hold(intent).map_err(|error| match error {
            HoldError::Full => SubmitError::Full {
                capacity: crate::destination::MAX_HELD_RELEASES,
            },
            HoldError::NoPeriod => {
                SubmitError::MalformedIntent("committee update without a period".into())
            }
        })?;
        let _ = self.events.send(MempoolEvent::DestinationObjectAdmitted {
            release: match admitted {
                DestinationAdmitted::Release { n } | DestinationAdmitted::ReleaseOverBase { n } => {
                    Some(n)
                }
                DestinationAdmitted::CommitteeUpdate { .. } => None,
            },
        });
        let _ = self.events.send(MempoolEvent::BridgeIntentAdmitted {
            intent_bytes: Arc::from(bytes),
        });
        tracing::info!(?admitted, "destination object admitted");
        Ok(admitted)
    }

    /// Forget the destination intents a template's block was refused for
    /// (its releases `releases`, the committee update bringing
    /// `committee_period`): they are not offered again unless submitted
    /// again.
    pub async fn forget_destination(&self, releases: &[u64], committee_period: Option<u64>) {
        let forgotten = self
            .state
            .lock()
            .await
            .destination
            .forget(releases, committee_period);
        if forgotten > 0 {
            tracing::warn!(
                ?releases,
                ?committee_period,
                forgotten,
                "destination intents forgotten: a template's block was refused for them"
            );
        }
    }

    /// The destination intents held for the template: committee updates by
    /// increasing period, releases by increasing `n`.
    pub async fn destination_candidates(
        &self,
    ) -> (
        Vec<crate::destination::CommitteeUpdateIntent>,
        Vec<crate::destination::ReleaseIntent>,
    ) {
        let st = self.state.lock().await;
        (
            st.destination.committee_updates().cloned().collect(),
            st.destination.releases().cloned().collect(),
        )
    }

    // -----------------------------------------------------------------------
    // v1.7 guardian objects (T3.7)
    // -----------------------------------------------------------------------

    /// Admit one encoded guardian intent ([`crate::guardian`]) handed in by
    /// `source` (a peer, an RPC client). In this order: the bytes; for an
    /// attestation or an equivocation, the rate limit of `source` (beyond
    /// it, refused unverified); under the lock, the v1.7 clock of the next
    /// block, the conflicts with what is held, every guardian rule but the
    /// signatures on the tip's guardian state after the held objects, and the
    /// transaction against the current state; outside the lock, the BLS
    /// signatures, then the transaction's authorization; under the lock
    /// again, the same cheap checks before it is held. Never in the fee pool,
    /// and no miner is woken: no guardian object is urgent (T1.1 §10).
    pub async fn submit_guardian(
        &self,
        bytes: &[u8],
        source: &str,
    ) -> Result<crate::guardian::GuardianAdmitted, SubmitError> {
        use crate::guardian::{group_facts, verify_guardian_signatures, GuardianIntent};
        use jetsam_chain::consensus::guardian::GuardianObjectKind;
        let intent = GuardianIntent::decode(bytes)
            .map_err(|error| SubmitError::MalformedIntent(format!("guardian intent: {error}")))?;
        let signed = matches!(
            intent.object.kind,
            GuardianObjectKind::Attestation | GuardianObjectKind::Equivocation
        );
        if signed
            && !self
                .state
                .lock()
                .await
                .guardian
                .limiter
                .admit(source, std::time::Instant::now())
        {
            return Err(SubmitError::GuardianRateLimited {
                origin: source.to_string(),
            });
        }
        let spend = validate_paged_spend(&intent.payment.pages)
            .map_err(|e| SubmitError::MalformedIntent(format!("PagedSpend: {e}")))?;
        if intent.payment.authorization_bytes.is_empty() {
            return Err(SubmitError::MissingProof);
        }
        if intent.payment.authorization_bytes.len() > MAX_AUTHORIZATION_BYTES {
            return Err(SubmitError::ProofTooLarge {
                actual: intent.payment.authorization_bytes.len(),
                max: MAX_AUTHORIZATION_BYTES,
            });
        }
        {
            let st = self.state.lock().await;
            check_guardian_admission(&intent, &spend, &st)?;
        }
        if signed {
            let object = intent.object.clone();
            let rules = self.config.guardian_rules;
            self.guardian_signature_checks
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tokio::task::spawn_blocking(move || verify_guardian_signatures(&object, &rules))
                .await
                .map_err(|e| SubmitError::Internal(format!("spawn_blocking: {e}")))?
                .map_err(guardian_submit_error)?;
        }
        let proof_bytes = intent.payment.authorization_bytes.clone();
        let pages = intent.payment.pages.clone();
        let executor = Arc::clone(&self.auth_verify_executor);
        {
            let _permit =
                self.auth_verify_semaphore.acquire().await.map_err(|_| {
                    SubmitError::Internal("proof-verification semaphore closed".into())
                })?;
            tokio::task::spawn_blocking(move || {
                executor(Box::new(move || {
                    verify_intent_authorization(&pages, &proof_bytes)
                }))
            })
            .await
            .map_err(|e| SubmitError::Internal(format!("spawn_blocking: {e}")))?
            .map_err(SubmitError::InvalidProof)?;
        }
        let mut st = self.state.lock().await;
        let pending_pair = check_guardian_admission(&intent, &spend, &st)?;
        let kind = intent.object.kind;
        let marker = intent.object.marker();
        let group = group_facts(&intent.payment.pages, spend.input_owner, spend.fee);
        let seq = st
            .guardian
            .hold(intent, group)
            .map_err(|refusal| SubmitError::GuardianNotHeld(refusal.to_string()))?;
        let _ = self.events.send(MempoolEvent::BridgeIntentAdmitted {
            intent_bytes: Arc::from(bytes),
        });
        tracing::info!(?kind, seq, pending_pair, "guardian object admitted");
        Ok(crate::guardian::GuardianAdmitted {
            kind,
            marker,
            seq,
            pending_pair,
        })
    }

    /// The guardian intents held for the template, in its order: the
    /// equivocation, the attestation, the delegation objects by arrival.
    pub async fn guardian_candidates(&self) -> Vec<crate::guardian::HeldGuardian> {
        self.state.lock().await.guardian.held().cloned().collect()
    }

    /// Forget the guardian intents numbered `seqs` (a template's block was
    /// refused for them): not offered again unless submitted again.
    pub async fn forget_guardian(&self, seqs: &[u64]) {
        let forgotten = self.state.lock().await.guardian.forget(seqs);
        if forgotten > 0 {
            tracing::warn!(
                ?seqs,
                forgotten,
                "guardian intents forgotten: a template's block was refused for them"
            );
        }
    }

    /// The v1.7 guardian rules this pool judges by.
    pub fn guardian_rules(&self) -> jetsam_chain::consensus::guardian::GuardianRules {
        self.config.guardian_rules
    }

    /// Guardian BLS verifications this pool has run.
    pub fn guardian_signature_checks(&self) -> u64 {
        self.guardian_signature_checks
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    // -----------------------------------------------------------------------
    // Block assembly
    // -----------------------------------------------------------------------

    /// Select fee-ordered indivisible groups fitting `max_pages` pages.
    ///
    /// Returns a fee-sorted list of `(Transaction, Option<cached_proof>)`.
    /// The caller (block builder) applies conflict resolution and coinbase on top.
    ///
    /// Returned txs are in descending fee-rate order with txid tie-break.
    pub async fn select_for_block(&self, max_pages: usize) -> Vec<SelectedMempoolEntry> {
        let st = self.state.lock().await;
        let limit = max_pages.min(BLOCK_MAX_USER_PAGES);
        st.pool
            .select_for_block(limit)
            .into_iter()
            .map(|entry| SelectedMempoolEntry {
                pages: entry.pages.clone(),
                logical_txid: entry.spend.logical_txid,
                cached_authorization: entry.cached_authorization().map(<[u8]>::to_vec),
            })
            .collect()
    }

    /// Select the same fee-ordered current-anchor prefix used by block
    /// assembly, cloning only the entries the caller can actually prove.
    ///
    /// The scan remains bounded by the consensus block maximum. Filtering is
    /// performed while entries are borrowed under the pool lock, so a
    /// memory-governed B25 template does not first clone up to 255 cached proof
    /// bundles and discard the excess.
    pub async fn select_for_block_at_anchor(
        &self,
        max_pages: usize,
        epoch_anchor: [u8; 32],
    ) -> Vec<SelectedMempoolEntry> {
        let st = self.state.lock().await;
        let limit = max_pages.min(BLOCK_MAX_USER_PAGES);
        st.pool
            .select_for_block_at_anchor(limit, &epoch_anchor)
            .into_iter()
            .map(|entry| SelectedMempoolEntry {
                pages: entry.pages.clone(),
                logical_txid: entry.spend.logical_txid,
                cached_authorization: entry.cached_authorization().map(<[u8]>::to_vec),
            })
            .collect()
    }

    /// [`Self::select_for_block_at_anchor`] over the pair of anchors a
    /// generation accepts; the launch generation consults only the current
    /// one.
    pub async fn select_for_block_at_anchors(
        &self,
        max_pages: usize,
        accepted: jetsam_chain::consensus::AcceptedEpochAnchors,
        generation: jetsam_chain::consensus::params::HistoryStepPackGeneration,
    ) -> Vec<SelectedMempoolEntry> {
        let st = self.state.lock().await;
        let limit = max_pages.min(BLOCK_MAX_USER_PAGES);
        st.pool
            .select_for_block_at_anchors(limit, accepted, generation)
            .into_iter()
            .map(|entry| SelectedMempoolEntry {
                pages: entry.pages.clone(),
                logical_txid: entry.spend.logical_txid,
                cached_authorization: entry.cached_authorization().map(<[u8]>::to_vec),
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Block confirmation
    // -----------------------------------------------------------------------

    /// Called when a new block is confirmed. Updates the chain view, removes
    /// confirmed txs, evicts expired txs, and broadcasts events.
    ///
    /// `confirmed_hashes`: derived txids of all txs in the confirmed block.
    /// `new_view`: updated chain state snapshot.
    pub async fn on_new_block(
        &self,
        confirmed_hashes: &[TxBodyHash],
        new_height: u64,
        new_view: ChainView,
    ) {
        let mut st = self.state.lock().await;

        // M-1 (audit 2026-11-10): the floor learns what blocks paid — the
        // rates of this pool's transactions the block carried — never what
        // was merely admitted.
        for hash in confirmed_hashes {
            if let Some(fee_rate) = st.pool.get(hash).map(|entry| entry.fee_rate) {
                st.floor.record_mined(fee_rate);
            }
        }

        // Remove confirmed txs.
        let removed = st.pool.on_block_confirmed(confirmed_hashes);
        for &hash in confirmed_hashes {
            let _ = self.events.send(MempoolEvent::TxConfirmed {
                hash,
                block_height: new_height,
            });
        }

        // Update chain view BEFORE eviction so anchor check uses new state.
        st.view = new_view;

        // Evict transactions whose anchor the next block's generation no
        // longer accepts. At launch that is every anchor but the current one;
        // from v1.3 a transaction bound to the previous epoch survives its
        // own boundary, which is the whole point of the second anchor.
        let (accepted_anchors, generation) = st.view.accepted_user_epoch_anchors();
        let stale_anchor: Vec<TxBodyHash> = st
            .pool
            .iter()
            .filter(|(_, entry)| !accepted_anchors.accepts(entry.spend.epoch_anchor, generation))
            .map(|(hash, _)| *hash)
            .collect();
        for hash in stale_anchor {
            st.pool.remove(&hash);
            let _ = self.events.send(MempoolEvent::TxEvicted {
                hash,
                reason: EvictReason::EpochAnchorChanged,
            });
        }

        // Evict txs whose output slots became occupied in the new block.
        // This happens when a coinbase (from a block mined while the tx was
        // in the mempool) landed on the same slot the wallet chose for its output.
        // The wallet must re-prove with fresh slot hints.
        use jetsam_chain::fri_state::SlotValue;
        let output_conflicts: Vec<TxBodyHash> = st
            .pool
            .iter()
            .filter_map(|(hash, entry)| {
                let occupied = entry.pages.iter().any(|page| {
                    page.body.live_outputs().any(|(_, out)| {
                        st.view
                            .try_slot(out.slot_index)
                            .map_or(true, |slot| slot != SlotValue::EMPTY)
                    })
                });
                if occupied {
                    Some(*hash)
                } else {
                    None
                }
            })
            .collect();
        for hash in output_conflicts {
            st.pool.remove(&hash);
            tracing::debug!(?hash, "tx evicted: output slot occupied by confirmed block");
            let _ = self.events.send(MempoolEvent::TxEvicted {
                hash,
                reason: EvictReason::OutputSlotOccupied,
            });
        }

        // Evict txs whose INPUT slots are no longer live in the new state.
        //
        // After a block is applied, some input slots of pool txs may have been
        // spent by other confirmed txs (not the same tx). Those pool txs are now
        // invalid: their input slot is EMPTY (was moved elsewhere by the block).
        //
        // They also fail silently in build_block_template (apply_tx returns Err),
        // wasting template-build cycles.
        let input_consumed: Vec<TxBodyHash> = st
            .pool
            .iter()
            .filter_map(|(hash, entry)| {
                let stale = entry.pages.iter().any(|page| {
                    page.body.live_inputs().any(|(_, inp)| {
                        // Input must still hold exactly (value, creation_id,
                        // group owner) for this intent to remain includable.
                        let expected = SlotValue::with_owner_fields(
                            inp.amount,
                            inp.creation_id,
                            entry.spend.input_owner.as_fields(),
                        );
                        st.view
                            .try_slot(inp.slot_index)
                            .map_or(true, |slot| slot != expected)
                    })
                });
                if stale {
                    Some(*hash)
                } else {
                    None
                }
            })
            .collect();
        let input_evict_count = input_consumed.len();
        for hash in input_consumed {
            st.pool.remove(&hash);
            tracing::debug!(
                ?hash,
                "tx evicted: input slot consumed or changed by confirmed block"
            );
            let _ = self.events.send(MempoolEvent::TxEvicted {
                hash,
                reason: EvictReason::InputConsumed,
            });
        }
        if input_evict_count > 0 {
            tracing::debug!(
                evicted = input_evict_count,
                "evicted stale txs with consumed input slots"
            );
        }

        // v1.5: what the next block may not carry under the client-object
        // rule of its height, re-judged at every view change (crossing J,
        // reorg): admission judged it for an earlier next block.
        let next_height = st.view.tip_height.saturating_add(1);
        evict_client_object_violations(&mut st, next_height, &self.events);
        // v1.7: likewise for the vault rule (crossing J, reorg).
        evict_vault_violations(&mut st, next_height, &self.events);
        evict_advance_violations(&mut st, next_height, &self.events);
        evict_guardian_violations(&mut st, next_height, &self.events);
        prune_destination(&mut st, next_height);
        prune_guardian(&mut st, next_height);

        // Rebuild slot sets after bulk eviction (O(pool) once/block vs O(N²) per submit).
        rebuild_slot_sets(&mut st);
        if st.pool.is_empty() {
            st.floor.reset();
        }

        tracing::debug!(
            height = new_height,
            confirmed = confirmed_hashes.len(),
            removed_from_pool = removed,
            pool_size = st.pool.len(),
            "mempool updated after new block"
        );
    }

    /// Re-admit transactions that were reclaimed by a chain reorg.
    ///
    /// These TXs were in reverted blocks. We log the count for observability
    /// and evict any that happen to be sitting in the pool already (duplicate
    /// re-submission race). Full re-admission with fresh authorizations is the
    /// wallet's responsibility — wallets detect the unconfirmed state via
    /// wallet scan and resubmit.
    ///
    /// NOTE: We do not have the original authorization bytes after a reorg (they
    /// are not persisted). Durable TX storage could enable
    /// automatic re-admission without wallet involvement.
    pub async fn readmit_after_reorg(&self, tx_hashes: Vec<TxBodyHash>) {
        if tx_hashes.is_empty() {
            return;
        }

        tracing::info!(
            count = tx_hashes.len(),
            "reorg: {} TX(s) reclaimed — wallets should resubmit if needed",
            tx_hashes.len()
        );

        // Evict any entries with the same hash that may have been re-submitted
        // concurrently (unlikely but keeps the pool clean).
        let mut st = self.state.lock().await;
        for hash in &tx_hashes {
            if st.pool.contains(hash) {
                st.pool.remove(hash);
                tracing::debug!(?hash, "reorg: removed re-submitted duplicate from pool");
            }
        }
        if st.pool.is_empty() {
            st.floor.reset();
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Number of transactions currently in the pool.
    pub async fn len(&self) -> usize {
        self.state.lock().await.pool.len()
    }

    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Current dynamic fee floor: a rate asked of each transaction for its
    /// weight ([`RelayFloor::required`]).
    pub async fn relay_floor(&self) -> RelayFloor {
        self.state.lock().await.floor.current()
    }

    /// Snapshot of output slots currently reserved by admitted mempool txs.
    pub async fn reserved_output_slots(&self) -> HashSet<u32> {
        self.state.lock().await.admitted_output_slots.clone()
    }

    /// Snapshot of input slots currently reserved by admitted mempool txs.
    pub async fn reserved_input_slots(&self) -> HashSet<u32> {
        self.state.lock().await.admitted_input_slots.clone()
    }

    /// Atomically snapshot both input and output reservations.
    ///
    /// Wallet reloads use this instead of two independent reads so an
    /// intervening admission/eviction cannot leave their pending sets sourced
    /// from different mempool states.
    pub async fn reserved_slots(&self) -> (HashSet<u32>, HashSet<u32>) {
        let state = self.state.lock().await;
        (
            state.admitted_input_slots.clone(),
            state.admitted_output_slots.clone(),
        )
    }

    /// Current chain occupancy used for fee estimation: (active slots, log_slots).
    pub async fn fee_context(&self) -> (u64, u32) {
        let st = self.state.lock().await;
        (st.view.active_slot_count, st.view.log_slots())
    }

    /// Pending value sent to an external owner, excluding change from that
    /// same owner. The underlying index is maintained at admission/removal.
    pub async fn pending_incoming_for_owner(&self, owner: &Address) -> u64 {
        self.state
            .lock()
            .await
            .pool
            .pending_incoming_for_owner(owner)
    }

    /// Snapshot compact RPC metadata without cloning intent/proof payloads.
    pub async fn metadata_snapshot(&self) -> MempoolMetadataSnapshot {
        let st = self.state.lock().await;
        MempoolMetadataSnapshot {
            fee_floor: st.floor.current(),
            entries: st
                .pool
                .iter()
                .map(|(hash, entry)| entry_metadata(*hash, entry))
                .collect(),
        }
    }

    /// Read count and retained-byte pressure without cloning entry metadata or
    /// any transaction/proof payload.
    pub async fn usage_snapshot(&self) -> MempoolUsageSnapshot {
        let state = self.state.lock().await;
        MempoolUsageSnapshot {
            size: state.pool.len(),
            capacity: self.config.capacity,
            intent_bytes: state.pool.total_intent_bytes(),
            max_intent_bytes: self.config.max_total_intent_bytes,
            fee_floor: state.floor.current(),
        }
    }

    /// O(1) compact lookup without cloning the entry's retained byte payloads.
    pub async fn get_entry_metadata(&self, hash: &TxBodyHash) -> Option<MempoolEntryMetadata> {
        let st = self.state.lock().await;
        st.pool.get(hash).map(|entry| entry_metadata(*hash, entry))
    }

    /// Clone at most one bounded mempool-sync response while holding one
    /// consistent pool lock.
    pub async fn intent_bytes_prefix(
        &self,
        max_txs: usize,
        max_total_bytes: usize,
        max_tx_bytes: usize,
    ) -> Vec<Vec<u8>> {
        let st = self.state.lock().await;
        st.pool
            .intent_bytes_prefix(max_txs, max_total_bytes, max_tx_bytes)
    }

    /// Update the chain view without applying a new block.
    /// Used on startup (initial state) or after a reorg.
    pub async fn update_chain_view(&self, view: ChainView) {
        let mut st = self.state.lock().await;
        st.view = view;
        let next_height = st.view.tip_height.saturating_add(1);
        let evicted = evict_client_object_violations(&mut st, next_height, &self.events)
            + evict_vault_violations(&mut st, next_height, &self.events)
            + evict_advance_violations(&mut st, next_height, &self.events)
            + evict_guardian_violations(&mut st, next_height, &self.events);
        prune_destination(&mut st, next_height);
        prune_guardian(&mut st, next_height);
        if evicted > 0 {
            rebuild_slot_sets(&mut st);
            if st.pool.is_empty() {
                st.floor.reset();
            }
        }
    }

    /// The client-object rule of the block at `next_height`
    /// (`check_plain_transaction`) and the v1.7 vault rule
    /// (`check_plain_vault_transaction`) applied to every entry, the refused
    /// ones evicted ([`EvictReason::ClientObjectRule`],
    /// [`EvictReason::VaultRule`]): a template builder calls it for the child
    /// of its own parent, which the pool's view may not have reached yet.
    /// Nothing below the v1.5 / v1.7 heights. Returns the number evicted.
    pub async fn evict_unminable_at(&self, next_height: u64) -> usize {
        let mut st = self.state.lock().await;
        let evicted = evict_client_object_violations(&mut st, next_height, &self.events)
            + evict_vault_violations(&mut st, next_height, &self.events)
            + evict_advance_violations(&mut st, next_height, &self.events)
            + evict_guardian_violations(&mut st, next_height, &self.events);
        if evicted > 0 {
            rebuild_slot_sets(&mut st);
            if st.pool.is_empty() {
                st.floor.reset();
            }
        }
        evicted
    }

    /// The v1.7 vault rules this pool judges plain transactions by.
    pub fn vault_rules(&self) -> jetsam_chain::consensus::vault::VaultRules {
        self.config.vault_rules
    }

    /// The v1.5 client-object rules this pool judges plain transactions by.
    pub fn client_object_rules(
        &self,
    ) -> jetsam_chain::consensus::client_objects::ClientObjectRules {
        self.config.client_object_rules
    }

    /// Serialized owner-auth proof bytes for the given admitted tx body
    /// hashes — the byte-exact objects this pool cryptographically verified
    /// at admission. Block acceptance uses them as a re-verification fast
    /// path: a block-carried proof that serializes to the same bytes needs
    /// no second verification; anything else falls back to the full check.
    pub async fn verified_authorization_proof_bytes(
        &self,
        hashes: &[TxBodyHash],
    ) -> std::collections::HashMap<[u8; 32], Vec<u8>> {
        use jetsam_gkr::WalletAuthorizationBundle;
        let mut out = std::collections::HashMap::with_capacity(hashes.len());
        for hash in hashes {
            // Clone only this one bounded bundle while holding the lock. The
            // raw intent (which contains the same authorization again) remains
            // resident in the pool and no all-entry byte snapshot is built.
            let authorization = {
                let state = self.state.lock().await;
                state
                    .pool
                    .get(hash)
                    .and_then(|entry| entry.cached_authorization().map(<[u8]>::to_vec))
            };
            let Some(authorization) = authorization else {
                continue;
            };
            let Ok(bundle) = WalletAuthorizationBundle::from_bytes(&authorization) else {
                continue;
            };
            let Some(proof_bytes) = bundle.proof_wire_bytes() else {
                continue;
            };
            out.insert(hash.0, proof_bytes);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Helper: all cheap admission checks
// ---------------------------------------------------------------------------

/// Bind the semantic object to its retained wire allocation without encoding
/// or cloning the potentially large authorization proof a second time.
fn canonical_intent_bytes_match(intent: &PagedSpendIntent, bytes: &[u8]) -> bool {
    let Ok(auth_len) = u32::try_from(intent.authorization_bytes.len()) else {
        return false;
    };
    let Ok(auth_offset) = intent.authorization_wire_offset() else {
        return false;
    };
    let Some(expected_len) = auth_offset.checked_add(auth_len as usize) else {
        return false;
    };
    if bytes.len() != expected_len {
        return false;
    }

    let mut prefix = Vec::with_capacity(auth_offset);
    prefix.push(PAGED_SPEND_INTENT_MARKER);
    prefix.extend_from_slice(&(intent.pages.len() as u16).to_le_bytes());
    for page in &intent.pages {
        if page.encode(&mut prefix).is_err() {
            return false;
        }
    }
    prefix.extend_from_slice(&auth_len.to_le_bytes());
    prefix.len() == auth_offset
        && bytes.starts_with(&prefix)
        && bytes[auth_offset..] == intent.authorization_bytes
}

/// Run every cheap admission check against `st`.
///
/// Called **twice** per `submit`:
/// - Pre-proof filter (DoS guard): rejects invalid txs before CPU-heavy work.
/// - Post-proof TOCTOU guard: final authority against current state.
///
/// Returns `anchor_height` (needed by `pool.admit` for expiry tracking).
/// The pre-filter discards it; the final admission step uses it.
fn run_admission_checks(
    pages: &[TxPage],
    spend: &PagedSpendFacts,
    st: &MempoolState,
) -> Result<u64, SubmitError> {
    run_admission_checks_with(pages, spend, st, Admission::Plain)
}

/// What a transaction is admitted as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Admission {
    /// A plain transaction of the fee pool: every pool rule.
    Plain,
    /// A transaction paying a client or destination object, which travels
    /// with its object and never enters the pool: no marker policy, no
    /// conflict with pool entries (the template builder leaves those out).
    ObjectPayment,
    /// A release group (T2.5): as an object payment, but its fee is capped
    /// by the vault gate at `R` (D59), so only the consensus minimum is
    /// asked, never the dynamic floor; its shape is
    /// `vault::check_release_group`'s.
    VaultRelease,
    /// A transaction paying a guardian object (T3.7): as an object payment,
    /// and the guardian rules are the object's (it may pay a pool).
    GuardianPayment,
    /// A guardian payout group (T3.7): as a guardian payment, its fee taken
    /// out of the entry and capped by `PAYOUT_FEE_MAX` (Y-6), so only the
    /// consensus minimum is asked, as for a release.
    GuardianPayout,
}

/// [`run_admission_checks`] for a transaction admitted as `admission`
/// ([`Admission`]).
fn run_admission_checks_with(
    pages: &[TxPage],
    spend: &PagedSpendFacts,
    st: &MempoolState,
    admission: Admission,
) -> Result<u64, SubmitError> {
    let pool_rules = admission == Admission::Plain;
    // Dynamic fee floor layered over the deterministic consensus minimum.
    let consensus_required = fee_breakdown(
        u64::from(spend.live_inputs),
        u64::from(spend.live_outputs),
        st.view.active_slot_count,
        st.view.log_slots(),
    )
    .required_total;
    let required = if matches!(admission, Admission::VaultRelease | Admission::GuardianPayout) {
        consensus_required
    } else {
        st.floor
            .current()
            .required(u64::from(spend.live_inputs), u64::from(spend.live_outputs))
            .max(consensus_required)
    };
    let actual = spend.fee;
    if actual < required {
        return Err(SubmitError::Consensus(
            jetsam_chain::consensus::ConsensusError::BelowMinFee { required, actual },
        ));
    }

    validate_paged_spend(pages)
        .map_err(|error| SubmitError::MalformedIntent(format!("PagedSpend: {error}")))?;
    // M-1 (audit 2026-11-10): a group the template builder never places — 256
    // touched segments beside the coinbase's — stayed here unmined, relayed to
    // every node, its fee their floor. Judged by the builder's own predicate,
    // before any state read.
    if pool_rules {
        let group: Vec<jetsam_tx::Transaction> = pages
            .iter()
            .map(|page| jetsam_tx::Transaction::new(page.body.clone()))
            .collect();
        if !jetsam_chain::consensus::template::group_fits_an_empty_template(&group, spend) {
            return Err(SubmitError::NoTemplateCarries(format!(
                "{} pages, {} inputs, {} outputs exceed one block's resources beside its coinbase",
                pages.len(),
                spend.live_inputs,
                spend.live_outputs
            )));
        }
    }
    // v1.5: a spend from the license addresses or a marker owner can never be
    // mined, and a transaction paying a client object is mined only with its
    // object (it does not travel as a plain transaction). Inert before v1.5.
    if pool_rules {
        jetsam_chain::consensus::client_objects::check_plain_transaction(
            pages,
            st.view.tip_height.saturating_add(1),
            &st.client_rules,
        )
        .map_err(|error| {
            SubmitError::Consensus(jetsam_chain::consensus::ConsensusError::ClientObject(error))
        })?;
        // v1.7: no spend from the vault, no release or committee-update
        // marker without its object, deposits of the deposit shape above the
        // bridge opening (V1, V4, V4-d), and V8 against the tip's vault.
        check_plain_vault_admission(pages, st, st.view.tip_height.saturating_add(1))
            .map_err(vault_submit_error)?;
    } else if let Some(owner) = pages
        .iter()
        .map(|page| page.body.input_owner)
        .find(jetsam_chain::consensus::client_objects::is_client_locked_address)
    {
        return Err(SubmitError::Consensus(
            jetsam_chain::consensus::ConsensusError::ClientObject(
                jetsam_chain::consensus::client_objects::ClientObjectError::SpendFromLockedAddress {
                    owner,
                },
            ),
        ));
    }
    // v1.7 (A2-M2): a transaction paying for an object other than a release
    // is held to the vault rules of a plain one (its own object's marker
    // aside): the vault gate judges its group like any other.
    if matches!(
        admission,
        Admission::ObjectPayment | Admission::GuardianPayment | Admission::GuardianPayout
    ) && st
        .vault_rules
        .active_at(st.view.tip_height.saturating_add(1))
    {
        crate::destination::check_object_payment_vault(pages, &st.vault_rules)
            .map_err(vault_submit_error)?;
    }
    // v1.7, the return over Base: an advance or a transfer of its claim is
    // judged as the block judges its group, against the branch's advances
    // (D114); any other transaction carrying an advance marker is refused.
    check_advance_admission(pages, spend, st).map_err(destination_submit_error)?;
    // v1.7: a transaction that does not travel with its guardian object
    // never touches the two pools and carries no guardian marker.
    if !matches!(admission, Admission::GuardianPayment | Admission::GuardianPayout) {
        crate::guardian::check_plain_guardian_transaction(
            pages,
            st.view.tip_height.saturating_add(1),
            &st.guardian_rules,
        )
        .map_err(guardian_submit_error)?;
    }
    if spend.epoch_anchor == [0u8; 32] {
        return Err(SubmitError::Consensus(
            jetsam_chain::consensus::ConsensusError::BadEpochAnchor,
        ));
    }

    // Epoch anchor is one the next block's generation accepts: the single
    // start-of-next-block anchor at launch, either of the two from v1.3.
    // Returns the current anchor's height for deterministic bookkeeping.
    let anchor_height = tx_epoch_anchor_height_for_child(st.view.tip_height + 1);
    let (accepted_anchors, generation) = st.view.accepted_user_epoch_anchors();
    if st.view.user_epoch_anchor_id == [0u8; 32]
        || !accepted_anchors.accepts(spend.epoch_anchor, generation)
    {
        return Err(SubmitError::Consensus(
            jetsam_chain::consensus::ConsensusError::BadEpochAnchor,
        ));
    }

    // No slot conflict with currently admitted txs (O(inputs + outputs)).
    if pool_rules {
        check_slot_conflicts_with_pool(pages, &st.admitted_input_slots, &st.admitted_output_slots)?;
    }

    // Input slots must be live in state.
    check_input_slots(pages, spend, &st.view)?;

    // Output slots must be empty in state.
    check_output_slots(pages, &st.view)?;

    Ok(anchor_height)
}

// ---------------------------------------------------------------------------
// Helper: rebuild admitted slot sets from current pool (O(pool), after eviction)
// ---------------------------------------------------------------------------

/// Evict every entry the block at `next_height` may not carry as a plain
/// transaction under the v1.5 client-object rule of that height
/// (`check_plain_transaction`, the rule admission applies and block
/// validation refuses): a marker payment without its object, a spend from a
/// locked address. Inert below the v1.5 height, where nothing is scanned.
/// Returns the number evicted; the caller rebuilds the slot sets.
fn evict_client_object_violations(
    st: &mut MempoolState,
    next_height: u64,
    events: &broadcast::Sender<MempoolEvent>,
) -> usize {
    if !st.client_rules.active_at(next_height) {
        return 0;
    }
    let refused: Vec<TxBodyHash> = st
        .pool
        .iter()
        .filter(|(_, entry)| {
            jetsam_chain::consensus::client_objects::check_plain_transaction(
                &entry.pages,
                next_height,
                &st.client_rules,
            )
            .is_err()
        })
        .map(|(hash, _)| *hash)
        .collect();
    for hash in &refused {
        st.pool.remove(hash);
        tracing::info!(
            ?hash,
            next_height,
            "tx evicted: no block from the v1.5 height may carry it as a plain transaction"
        );
        let _ = events.send(MempoolEvent::TxEvicted {
            hash: *hash,
            reason: EvictReason::ClientObjectRule,
        });
    }
    refused.len()
}

/// The v1.7 vault rules a plain transaction meets at admission for the block
/// at `next_height`: [`check_plain_vault_transaction`] (V1, V4, V4-d, markers
/// without objects), then V8 against the vault of the pool's view (the
/// template applies V8 again in block order). Inert below J.
///
/// [`check_plain_vault_transaction`]: jetsam_chain::consensus::vault::check_plain_vault_transaction
fn check_plain_vault_admission(
    pages: &[TxPage],
    st: &MempoolState,
    next_height: u64,
) -> Result<(), jetsam_chain::consensus::vault::VaultError> {
    use jetsam_chain::consensus::vault::{check_plain_vault_transaction, check_vault_deposit_room};
    check_plain_vault_transaction(pages, next_height, &st.vault_rules)?;
    if st.vault_rules.active_at(next_height) {
        if let Some(vault) = &st.view.vault {
            check_vault_deposit_room(pages, vault.bridge_locked, &st.vault_rules)?;
        }
    }
    Ok(())
}

fn vault_submit_error(error: jetsam_chain::consensus::vault::VaultError) -> SubmitError {
    SubmitError::Consensus(jetsam_chain::consensus::ConsensusError::DestinationBlock(
        jetsam_chain::consensus::destination::state::DestinationBlockError::Vault(error),
    ))
}

/// The live outputs of a transaction's group, in order.
fn group_outputs(pages: &[TxPage]) -> Vec<(jetsam_poseidon2b::primitives::Address, u64)> {
    pages
        .iter()
        .flat_map(|page| {
            page.body
                .live_outputs()
                .map(|(_, output)| (output.owner, output.amount))
        })
        .collect()
}

/// The return over Base (D107, D114) for one transaction offered for the
/// block after the view's tip: from the return-over-Base height on, the rule
/// of the block's advance groups (`AdvanceCheck`) on its group, against the
/// view's advances and paid set — the same checks, so that no admitted group
/// can make a template refused. What depends on the other groups of the
/// block (a burn advanced twice, the cap over several, a transfer twice, the
/// release of `n` beside it) is the template's (`advances_within_rules`).
/// Inert below that height (every profile today).
fn check_advance_admission(
    pages: &[TxPage],
    spend: &PagedSpendFacts,
    st: &MempoolState,
) -> Result<(), jetsam_chain::consensus::destination::state::DestinationBlockError> {
    use jetsam_chain::consensus::destination::advance::{carries_advance_marker, AdvanceCheck};
    use jetsam_chain::consensus::destination::state::DestinationBlockError;
    let next_height = st.view.tip_height.saturating_add(1);
    let base_return = &st.view.destination_rules.base_return;
    if !base_return.active_at(next_height) {
        return Ok(());
    }
    let outputs = group_outputs(pages);
    if !carries_advance_marker(&outputs) {
        return Ok(());
    }
    let Some(vault) = st.view.vault.as_ref() else {
        return Err(DestinationBlockError::StateNotParent {
            tip: st.view.tip_height,
            height: next_height,
        });
    };
    AdvanceCheck::new(
        &vault.advances,
        &vault.released,
        Vec::new(),
        &st.vault_rules,
        &base_return.advance,
    )
    .group(0, spend.input_owner, &outputs)
    .map_err(DestinationBlockError::Advance)
}

/// Evict every entry that carries an advance marker and that the block at
/// `next_height` can never carry as it is (from the return-over-Base height
/// on): a refusal of its own group, or one the branch makes final (its burn
/// paid or advanced already, a marker taken). A transfer whose advance or
/// creditor is still to come, and an advance over its advancer's cap (it fits
/// once an advance is consumed), stay: the template leaves them out
/// meanwhile. Inert below that height. Returns the number evicted.
fn evict_advance_violations(
    st: &mut MempoolState,
    next_height: u64,
    events: &broadcast::Sender<MempoolEvent>,
) -> usize {
    use jetsam_chain::consensus::destination::advance::AdvanceError;
    use jetsam_chain::consensus::destination::state::DestinationBlockError;
    if !st
        .view
        .destination_rules
        .base_return
        .active_at(next_height)
    {
        return 0;
    }
    let refused: Vec<TxBodyHash> = st
        .pool
        .iter()
        .filter(|(_, entry)| {
            let Ok(spend) = validate_paged_spend(&entry.pages) else {
                return false;
            };
            match check_advance_admission(&entry.pages, &spend, st) {
                Ok(()) => false,
                Err(DestinationBlockError::Advance(
                    AdvanceError::OutstandingCapReached { .. }
                    | AdvanceError::NoAdvanceToTransfer { .. }
                    | AdvanceError::TransferNotByCreditor { .. },
                )) => false,
                Err(_) => true,
            }
        })
        .map(|(hash, _)| *hash)
        .collect();
    for hash in &refused {
        st.pool.remove(hash);
        tracing::info!(
            ?hash,
            next_height,
            "tx evicted: no block from the return-over-Base height may carry this advance"
        );
        let _ = events.send(MempoolEvent::TxEvicted {
            hash: *hash,
            reason: EvictReason::AdvanceRule,
        });
    }
    refused.len()
}

/// Evict every entry the block at `next_height` may not carry as a plain
/// transaction under the v1.7 vault rule of that height
/// (`check_plain_vault_transaction`): a spend from the vault, a release or
/// committee-update marker without its object, a deposit of the wrong shape
/// or at or below the bridge opening. Inert below J. V8 does not evict: a
/// deposit that does not fit now may fit after a release (the template skips
/// it meanwhile). Returns the number evicted; the caller rebuilds the slot
/// sets.
fn evict_vault_violations(
    st: &mut MempoolState,
    next_height: u64,
    events: &broadcast::Sender<MempoolEvent>,
) -> usize {
    if !st.vault_rules.active_at(next_height) {
        return 0;
    }
    let refused: Vec<TxBodyHash> = st
        .pool
        .iter()
        .filter(|(_, entry)| {
            jetsam_chain::consensus::vault::check_plain_vault_transaction(
                &entry.pages,
                next_height,
                &st.vault_rules,
            )
            .is_err()
        })
        .map(|(hash, _)| *hash)
        .collect();
    for hash in &refused {
        st.pool.remove(hash);
        tracing::info!(
            ?hash,
            next_height,
            "tx evicted: no block from the v1.7 height may carry it as a plain transaction"
        );
        let _ = events.send(MempoolEvent::TxEvicted {
            hash: *hash,
            reason: EvictReason::VaultRule,
        });
    }
    refused.len()
}

/// What a destination intent is verified against beside the vault gate
/// (A2-M1): the committees the branch of the next block knows, the
/// destination rules and the committee keys of the pool's view.
pub(crate) struct DestinationJudge {
    committees: Vec<jetsam_chain::consensus::destination::state::CommitteeRef>,
    rules: jetsam_chain::consensus::destination::state::DestinationRules,
    keys: Arc<jetsam_chain::consensus::destination::state::CommitteeKeyCache>,
}

impl DestinationJudge {
    /// [`crate::destination::verify_destination_intent`] by this judge.
    fn verify(&self, intent: &crate::destination::DestinationIntent) -> Result<(), SubmitError> {
        crate::destination::verify_destination_intent(
            intent,
            &self.committees,
            &self.rules,
            &self.keys,
        )
        .map_err(destination_submit_error)
    }

    /// Whether `other` gives every intent the same verdict: the same rules
    /// and committees, the keys of each held by both or by neither (a key set
    /// is the one its root commits to).
    fn judges_like(&self, other: &Self) -> bool {
        self.rules == other.rules
            && self.committees == other.committees
            && self
                .committees
                .iter()
                .all(|c| self.keys.get(&c.root).is_some() == other.keys.get(&c.root).is_some())
    }
}

/// The admission rules of a destination intent against `st` but its L1
/// rules (module docs of [`crate::destination`]); returns the judge of
/// those.
fn check_destination_admission(
    intent: &crate::destination::DestinationIntent,
    spend: &PagedSpendFacts,
    st: &MempoolState,
) -> Result<DestinationJudge, SubmitError> {
    use crate::destination::DestinationIntent;
    use jetsam_chain::consensus::destination::advance::{advance_marker_kind, release_payee};
    use jetsam_chain::consensus::destination::state::DestinationBlockError;
    use jetsam_chain::consensus::destination::DestinationError;
    use jetsam_chain::consensus::vault::{check_release_group_paying, VaultError};
    let next_height = st.view.tip_height.saturating_add(1);
    let before_j = || {
        SubmitError::Consensus(jetsam_chain::consensus::ConsensusError::Destination(
            DestinationError::SectionBeforeActivation,
        ))
    };
    if !st.vault_rules.active_at(next_height) {
        return Err(before_j());
    }
    let vault = st.view.vault.as_ref().ok_or_else(before_j)?;
    match intent {
        DestinationIntent::CommitteeUpdate(update) => {
            if spend.input_owner == st.vault_rules.vault {
                return Err(vault_submit_error(VaultError::SpendFromLockedAddress {
                    group: 0,
                }));
            }
            run_admission_checks_with(&update.payment.pages, spend, st, Admission::ObjectPayment)?;
        }
        DestinationIntent::Release(release) => {
            if vault.released.contains(release.release.n) {
                return Err(vault_submit_error(VaultError::ReleaseAlreadyPaid {
                    n: release.release.n,
                }));
            }
            // From the return-over-Base height on a release is proven over
            // Base (objects 5 and 6), never before; and it pays the creditor
            // of the advance of exactly its burn, if the branch holds one
            // (D107 V2, D114). A burn owned by an advance marker is never
            // payable over Base (the advance rule refuses its group).
            let over_base = st.view.destination_rules.base_return.active_at(next_height);
            let payee = match (over_base, &release.base) {
                (false, None) => release.release.owner,
                (false, Some(_)) => {
                    return Err(destination_submit_error(
                        DestinationBlockError::BaseObjectBeforeActivation,
                    ))
                }
                (true, None) => {
                    return Err(destination_submit_error(
                        DestinationBlockError::BaseProofMissing,
                    ))
                }
                (true, Some(_)) => {
                    if advance_marker_kind(&release.release.owner).is_some() {
                        return Err(vault_submit_error(VaultError::ReleaseShape { group: 0 }));
                    }
                    release_payee(
                        &vault.advances,
                        release.release.n,
                        &release.release.owner,
                        release.release.amount,
                    )
                }
            };
            check_release_group_paying(
                &release.payment.pages,
                &release.release,
                &st.vault_rules,
                vault.floor,
                &payee,
            )
            .map_err(vault_submit_error)?;
            run_admission_checks_with(&release.payment.pages, spend, st, Admission::VaultRelease)?;
        }
    }
    Ok(DestinationJudge {
        committees: vault.committees.clone(),
        // The next block's: from the return-over-Base height on, the light
        // client reads the L1 Base settles on.
        rules: st.view.destination_rules.effective_at(next_height),
        keys: Arc::clone(&st.view.committee_keys),
    })
}

fn destination_submit_error(
    error: jetsam_chain::consensus::destination::state::DestinationBlockError,
) -> SubmitError {
    SubmitError::Consensus(jetsam_chain::consensus::ConsensusError::DestinationBlock(
        error,
    ))
}

/// Forget the destination intents the next block can no longer use: all of
/// them below J; above, releases whose index the branch paid or whose vault
/// slots are no longer live as their group spends them, and committee updates
/// for a period the branch already knows.
fn prune_destination(st: &mut MempoolState, next_height: u64) {
    use jetsam_chain::fri_state::SlotValue;
    if !st.vault_rules.active_at(next_height) {
        st.destination.clear();
        return;
    }
    let Some(vault) = st.view.vault.clone() else {
        return;
    };
    let view = &st.view;
    let over_base = view.destination_rules.base_return.active_at(next_height);
    let forgotten = st.destination.prune(
        |n| vault.released.contains(n),
        |release| {
            // Proven for the other side of the return-over-Base height, or
            // paying another than V2 now asks (an advance made meanwhile).
            if over_base != release.base.is_some()
                || (over_base
                    && release.paid_to()
                        != Some(
                            jetsam_chain::consensus::destination::advance::release_payee(
                                &vault.advances,
                                release.release.n,
                                &release.release.owner,
                                release.release.amount,
                            ),
                        ))
            {
                return true;
            }
            let Some(owner) = release
                .payment
                .pages
                .first()
                .map(|page| page.body.input_owner)
            else {
                return true;
            };
            release.payment.pages.iter().any(|page| {
                page.body.live_inputs().any(|(_, input)| {
                    let expected = SlotValue::with_owner_fields(
                        input.amount,
                        input.creation_id,
                        owner.as_fields(),
                    );
                    view.try_slot(input.slot_index)
                        .map_or(true, |slot| slot != expected)
                })
            })
        },
        vault.committees.last().map(|committee| committee.period),
    );
    if forgotten > 0 {
        tracing::debug!(forgotten, "destination intents the branch no longer needs");
    }
}

fn guardian_submit_error(error: jetsam_chain::consensus::guardian::GuardianError) -> SubmitError {
    SubmitError::Consensus(jetsam_chain::consensus::ConsensusError::Guardian(error))
}

/// Evict every entry the block at `next_height` may not carry as a plain
/// transaction under the v1.7 guardian policy
/// ([`crate::guardian::check_plain_guardian_transaction`]): crossing J, a
/// reorg. Inert below J. Returns the number evicted; the caller rebuilds the
/// slot sets.
fn evict_guardian_violations(
    st: &mut MempoolState,
    next_height: u64,
    events: &broadcast::Sender<MempoolEvent>,
) -> usize {
    use jetsam_chain::consensus::params::v1_7_active_with;
    if !v1_7_active_with(next_height, st.guardian_rules.activation_height) {
        return 0;
    }
    let refused: Vec<TxBodyHash> = st
        .pool
        .iter()
        .filter(|(_, entry)| {
            crate::guardian::check_plain_guardian_transaction(
                &entry.pages,
                next_height,
                &st.guardian_rules,
            )
            .is_err()
        })
        .map(|(hash, _)| *hash)
        .collect();
    for hash in &refused {
        st.pool.remove(hash);
        tracing::info!(
            ?hash,
            next_height,
            "tx evicted: no block from the v1.7 height may carry it without its guardian object"
        );
        let _ = events.send(MempoolEvent::TxEvicted {
            hash: *hash,
            reason: EvictReason::GuardianRule,
        });
    }
    refused.len()
}

/// The admission rules of a guardian intent against `st`, cheapest first
/// and without the BLS signatures (module docs of [`crate::guardian`]).
/// Returns whether it is a reward funding before the latch still waiting
/// for its pair (F-4).
fn check_guardian_admission(
    intent: &crate::guardian::GuardianIntent,
    spend: &PagedSpendFacts,
    st: &MempoolState,
) -> Result<bool, SubmitError> {
    use crate::guardian::{apply_guardian_intent, group_facts, project, Signatures};
    use jetsam_chain::consensus::guardian::{GuardianError, GuardianObjectKind, RewardFund};
    use jetsam_chain::consensus::params::v1_7_active_with;
    let rules = &st.guardian_rules;
    let next_height = st.view.tip_height.saturating_add(1);
    let state = st
        .view
        .guardian
        .as_deref()
        .filter(|_| v1_7_active_with(next_height, rules.activation_height))
        .ok_or_else(|| guardian_submit_error(GuardianError::GuardianObjectsBeforeV1_7))?;
    st.guardian
        .conflict(intent)
        .map_err(|refusal| SubmitError::GuardianNotHeld(refusal.to_string()))?;
    let group = group_facts(&intent.payment.pages, spend.input_owner, spend.fee);
    let (mut projected, _) = project(state, next_height, st.guardian.held(), &st.view, rules);
    apply_guardian_intent(
        &mut projected,
        next_height,
        &group,
        &intent.object,
        &st.view,
        rules,
        Signatures::Defer,
    )
    .map_err(guardian_submit_error)?;
    let admission = if intent.object.kind == GuardianObjectKind::Payout {
        Admission::GuardianPayout
    } else {
        Admission::GuardianPayment
    };
    run_admission_checks_with(&intent.payment.pages, spend, st, admission)?;
    let funder = |object: &jetsam_chain::consensus::guardian::GuardianObject| {
        (object.kind == GuardianObjectKind::RewardFund)
            .then(|| RewardFund::decode(&object.body).ok().map(|fund| fund.funder))
            .flatten()
    };
    Ok(match funder(&intent.object) {
        Some(funder) if state.activated_at.is_none() => {
            let other = if funder == rules.network_fund {
                rules.lab_fund
            } else {
                rules.network_fund
            };
            !st.guardian
                .held()
                .any(|held| funder_is(&held.intent.object, other))
        }
        _ => false,
    })
}

/// Whether `object` is a reward funding by `funder`.
fn funder_is(object: &jetsam_chain::consensus::guardian::GuardianObject, funder: Address) -> bool {
    object.kind == jetsam_chain::consensus::guardian::GuardianObjectKind::RewardFund
        && jetsam_chain::consensus::guardian::RewardFund::decode(&object.body)
            .is_ok_and(|fund| fund.funder == funder)
}

/// Judge the held guardian intents again for the block at `next_height`
/// (a new tip: a period boundary, `J_D` and the latch included) and forget
/// the ones the branch now refuses, and those whose inputs are no longer live
/// as their transaction spends them; everything below J or without a
/// guardian state.
fn prune_guardian(st: &mut MempoolState, next_height: u64) {
    use jetsam_chain::consensus::params::v1_7_active_with;
    let rules = st.guardian_rules;
    let Some(state) = st
        .view
        .guardian
        .clone()
        .filter(|_| v1_7_active_with(next_height, rules.activation_height))
    else {
        st.guardian.clear();
        return;
    };
    let (_, refused) = crate::guardian::project(
        &state,
        next_height,
        st.guardian.held(),
        &st.view,
        &rules,
    );
    let mut forget: Vec<u64> = refused.iter().map(|(seq, _)| *seq).collect();
    forget.extend(
        st.guardian
            .held()
            .filter(|held| inputs_no_longer_live(&held.intent.payment.pages, &st.view))
            .map(|held| held.seq),
    );
    let forgotten = st.guardian.forget(&forget);
    if forgotten > 0 {
        tracing::debug!(forgotten, ?refused, "guardian intents the branch no longer takes");
    }
}

/// Whether an input of `pages` is no longer the live slot it spends (value,
/// creation id and owner).
fn inputs_no_longer_live(pages: &[TxPage], view: &ChainView) -> bool {
    let Some(owner) = pages.first().map(|page| page.body.input_owner) else {
        return true;
    };
    pages.iter().any(|page| {
        page.body.live_inputs().any(|(_, input)| {
            let expected =
                SlotValue::with_owner_fields(input.amount, input.creation_id, owner.as_fields());
            view.try_slot(input.slot_index)
                .map_or(true, |slot| slot != expected)
        })
    })
}

fn rebuild_slot_sets(st: &mut MempoolState) {
    st.admitted_input_slots.clear();
    st.admitted_output_slots.clear();
    for (_, entry) in st.pool.iter() {
        for page in &entry.pages {
            for (_, inp) in page.body.live_inputs() {
                st.admitted_input_slots.insert(inp.slot_index);
            }
            for (_, out) in page.body.live_outputs() {
                st.admitted_output_slots.insert(out.slot_index);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: slot conflict with admitted pool — O(MAX_INPUTS + MAX_OUTPUTS)
// ---------------------------------------------------------------------------

fn check_slot_conflicts_with_pool(
    pages: &[TxPage],
    pool_inputs: &HashSet<u32>,
    pool_outputs: &HashSet<u32>,
) -> Result<(), SubmitError> {
    for page in pages {
        for (_, inp) in page.body.live_inputs() {
            if pool_inputs.contains(&inp.slot_index) {
                return Err(SubmitError::Consensus(
                    jetsam_chain::consensus::ConsensusError::SlotConflict,
                ));
            }
        }
        for (_, out) in page.body.live_outputs() {
            if pool_outputs.contains(&out.slot_index) {
                return Err(SubmitError::Consensus(
                    jetsam_chain::consensus::ConsensusError::SlotConflict,
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: input slots must be live in state
// ---------------------------------------------------------------------------

fn check_input_slots(
    pages: &[TxPage],
    spend: &PagedSpendFacts,
    view: &ChainView,
) -> Result<(), SubmitError> {
    for page in pages {
        for (_, inp) in page.body.live_inputs() {
            let idx = inp.slot_index;
            if (idx as u64) >= view.num_slots {
                return Err(SubmitError::Consensus(
                    jetsam_chain::consensus::ConsensusError::ShapeMismatch(format!(
                        "input slot {idx} out of range"
                    )),
                ));
            }
            let expected = SlotValue::with_owner_fields(
                inp.amount,
                inp.creation_id,
                spend.input_owner.as_fields(),
            );
            let actual = view.try_slot(idx).map_err(|error| {
                SubmitError::Internal(format!("chain state read failed: {error}"))
            })?;
            if actual != expected {
                tracing::warn!(
                    slot_index = idx,
                    expected_value = inp.amount,
                    expected_creation_id = inp.creation_id,
                    actual_empty = actual.is_empty(),
                    "check_input_slots: canonical slot mismatch"
                );
                return Err(SubmitError::Consensus(
                    jetsam_chain::consensus::ConsensusError::BadStateRoot,
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: output slots must be empty in state
// ---------------------------------------------------------------------------

fn check_output_slots(pages: &[TxPage], view: &ChainView) -> Result<(), SubmitError> {
    for page in pages {
        for (_, out) in page.body.live_outputs() {
            let idx = out.slot_index;
            if (idx as u64) >= view.num_slots {
                return Err(SubmitError::Consensus(
                    jetsam_chain::consensus::ConsensusError::ShapeMismatch(format!(
                        "output slot {idx} out of range"
                    )),
                ));
            }
            if view.try_slot(idx).map_err(|error| {
                SubmitError::Internal(format!("chain state read failed: {error}"))
            })? != SlotValue::EMPTY
            {
                return Err(SubmitError::Consensus(
                    jetsam_chain::consensus::ConsensusError::SlotConflict,
                ));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: Auth-only wallet authorization verification
// ---------------------------------------------------------------------------

/// Verify the wallet authorization for a non-coinbase tx.
/// Returns Ok(()) if valid, Err(String) with reason if invalid.
fn verify_intent_authorization(pages: &[TxPage], authorization_bytes: &[u8]) -> Result<(), String> {
    use jetsam_gkr::{verify_paged_spend_authorization, WalletAuthorizationBundle};

    let bundle = WalletAuthorizationBundle::from_bytes(authorization_bytes)
        .map_err(|e| format!("authorization decode: {e}"))?;
    verify_paged_spend_authorization(pages, &bundle)
        .map_err(|e| format!("authorization verify: {e}"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use jetsam_chain::consensus::genesis::genesis_header;
    use jetsam_chain::fri_state::SlotValue;
    use jetsam_chain::state::ChainState;
    use jetsam_poseidon2b::primitives::Address;
    use jetsam_tx::{
        output_bitmap_bit, validate_paged_spend, TxBody, TxInput, TxOutput, TxPage,
        PAGED_SPEND_END_BIT, PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
    };

    use super::{
        broadcast, check_input_slots, run_admission_checks, AsyncMempool, MempoolState,
        BLOCK_MAX_USER_PAGES, EVENT_CHANNEL_CAPACITY, MAX_MEMPOOL_TXS,
    };
    use crate::config::MempoolConfig;
    use crate::view::ChainView;
    use std::collections::HashSet;

    fn user_pages(epoch_anchor: [u8; 32], fee: u64, seed: u8) -> Vec<TxPage> {
        let mut inputs = [TxInput::dummy(); TX_INPUTS];
        inputs[0] = TxInput {
            slot_index: u32::from(seed) * 2 + 1,
            amount: 100 + fee,
            creation_id: u64::from(seed) + 1,
        };
        let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
        outputs[0] = TxOutput {
            slot_index: u32::from(seed) * 2 + 2,
            amount: 100,
            owner: Address([seed; 32]),
        };
        vec![TxPage::new(TxBody {
            epoch_anchor,
            fee,
            input_owner: Address([0xA5; 32]),
            inputs,
            outputs,
            validity_bitmap: 1 | output_bitmap_bit(0) | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT,
            is_coinbase: false,
        })
        .unwrap()]
    }

    fn retained_intent(auth_byte: u8, auth_len: usize) -> Vec<u8> {
        let mut bytes = vec![0; jetsam_tx::paged_spend_authorization_wire_offset(1).unwrap()];
        bytes.extend(std::iter::repeat_n(auth_byte, auth_len));
        bytes
    }

    #[tokio::test]
    async fn empty_pool_resets_stale_dynamic_fee_floor_on_block_advance() {
        use jetsam_chain::consensus::params::MIN_FEE_BASE;

        let state = ChainState::with_log_slots(6);
        let view = ChainView::new(0, HashMap::new(), 0, state.state);
        let pool = AsyncMempool::new(view.clone(), MempoolConfig::default());
        {
            let mut locked = pool.state.lock().await;
            for _ in 0..crate::floor::FEE_FLOOR_MIN_SAMPLES {
                locked.floor.record_mined(100_000);
            }
            assert!(locked.floor.current().required(1, 1) > MIN_FEE_BASE);
            assert!(locked.pool.is_empty());
        }

        pool.on_new_block(&[], 1, view).await;

        assert_eq!(pool.relay_floor().await.required(1, 1), MIN_FEE_BASE);
    }

    /// The worst burst is the whole pool evicted at a boundary while the block
    /// that caused it confirms its own transactions. Expressed against the
    /// constants that bound each half, so raising either one without raising
    /// the channel fails here rather than in production under load.
    #[test]
    fn the_event_channel_absorbs_a_whole_pool_evicted_beside_a_full_block() {
        let worst_burst = MAX_MEMPOOL_TXS + BLOCK_MAX_USER_PAGES + 1;
        assert!(
            EVENT_CHANNEL_CAPACITY >= worst_burst,
            "channel holds {EVENT_CHANNEL_CAPACITY}, one on_new_block can emit {worst_burst}"
        );

        // And the channel really does keep that many before dropping the
        // oldest — the property the arithmetic above is standing in for.
        let (tx, mut rx) = broadcast::channel::<u64>(EVENT_CHANNEL_CAPACITY);
        for value in 0..worst_burst as u64 {
            tx.send(value).expect("a live receiver");
        }
        assert!(
            matches!(rx.try_recv(), Ok(0)),
            "the oldest event was dropped, which is how a wallet loses a send"
        );
    }

    #[tokio::test]
    async fn a_transaction_left_behind_by_an_epoch_boundary_announces_why() {
        use crate::event::{EvictReason, MempoolEvent};

        let state = ChainState::with_log_slots(6);
        let mut view = ChainView::new(0, HashMap::new(), 0, state.state);
        let anchor = [0x11; 32];
        view.user_epoch_anchor_id = anchor;
        let pool = AsyncMempool::new(view.clone(), MempoolConfig::default());
        let pages = user_pages(anchor, 300, 1);
        let txid = validate_paged_spend(&pages).unwrap().logical_txid;
        {
            let mut locked = pool.state.lock().await;
            locked.pool.admit(pages, 0).expect("admit");
        }
        let mut events = pool.subscribe();

        // A boundary block installs a new anchor, and nothing built on the old
        // one can ever be mined. This is the ordinary fate of a transaction no
        // miner picked up within TX_EPOCH_BLOCKS, and it used to be silent.
        let mut after_boundary = view.clone();
        after_boundary.user_epoch_anchor_id = [0x22; 32];
        pool.on_new_block(&[], 1, after_boundary).await;

        let mut evicted = None;
        while let Ok(event) = events.try_recv() {
            if let MempoolEvent::TxEvicted { hash, reason } = event {
                evicted = Some((hash, reason));
            }
        }
        let (hash, reason) = evicted.expect("the pool must say what it dropped");
        assert_eq!(hash, txid);
        assert_eq!(reason, EvictReason::EpochAnchorChanged);
        assert!(
            reason.operator_explanation().contains("epoch"),
            "the words the operator reads must name the cause"
        );
    }

    #[tokio::test]
    async fn anchored_selection_filters_before_cloning_bounded_prefix() {
        let state = ChainState::with_log_slots(6);
        let pool = AsyncMempool::new(
            ChainView::new(0, HashMap::new(), 0, state.state),
            MempoolConfig::default().with_capacity(8),
        );
        let anchor = [0x11; 32];
        let wrong_anchor = [0x22; 32];
        let high = user_pages(anchor, 300, 1);
        let wrong = user_pages(wrong_anchor, 200, 2);
        let low = user_pages(anchor, 100, 3);
        let high_id = validate_paged_spend(&high).unwrap().logical_txid;
        let low_id = validate_paged_spend(&low).unwrap().logical_txid;
        {
            let mut locked = pool.state.lock().await;
            locked.pool.admit(high, 0).expect("admit high fee");
            locked.pool.admit(wrong, 0).expect("admit wrong anchor");
            locked.pool.admit(low, 0).expect("admit low fee");
            locked
                .pool
                .set_intent_bytes(&high_id, retained_intent(0xA5, 1024));
            locked
                .pool
                .set_intent_bytes(&low_id, retained_intent(0x5A, 1024));
        }

        let one = pool.select_for_block_at_anchor(1, anchor).await;
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].logical_txid, high_id);
        assert_eq!(one[0].cached_authorization.as_ref().unwrap().len(), 1024);

        let two = pool.select_for_block_at_anchor(2, anchor).await;
        assert_eq!(two.len(), 2);
        assert_eq!(two[0].logical_txid, high_id);
        assert_eq!(two[1].logical_txid, low_id);
    }

    #[tokio::test]
    async fn metadata_snapshot_never_carries_retained_intent_or_authorization_bytes() {
        let state = ChainState::with_log_slots(6);
        let pool = AsyncMempool::new(
            ChainView::new(0, HashMap::new(), 0, state.state),
            MempoolConfig::default().with_capacity(8),
        );
        let tx = user_pages([0x31; 32], 400, 1);
        let txid = validate_paged_spend(&tx).unwrap().logical_txid;
        {
            let mut locked = pool.state.lock().await;
            locked.pool.admit(tx, 7).expect("admit metadata fixture");
            locked
                .pool
                .set_intent_bytes(&txid, vec![0x5A; 2 * 1024 * 1024]);
        }

        let snapshot = pool.metadata_snapshot().await;
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].tx_hash, txid);
        assert!(snapshot.entries[0].has_authorization);
        assert!(std::mem::size_of::<super::MempoolEntryMetadata>() <= 64);

        let single = pool.get_entry_metadata(&txid).await.unwrap();
        assert_eq!(single, snapshot.entries[0]);
        let usage = pool.usage_snapshot().await;
        assert_eq!(usage.size, 1);
        assert_eq!(usage.capacity, 8);
        assert_eq!(usage.intent_bytes, 2 * 1024 * 1024);
        assert_eq!(
            usage.max_intent_bytes,
            jetsam_chain::consensus::wire_limits::MAX_MEMPOOL_BYTES
        );
        assert_eq!(
            pool.state.lock().await.pool.total_intent_bytes(),
            2 * 1024 * 1024
        );
    }

    #[test]
    fn template_and_preverified_paths_never_clone_raw_intent_payloads() {
        let source = include_str!("pool.rs");
        let selection = source
            .split("pub async fn select_for_block(")
            .nth(1)
            .expect("selection method")
            .split("// -----------------------------------------------------------------------\n    // Block confirmation")
            .next()
            .expect("selection boundary");
        assert!(!selection.contains("intent_bytes"));
        assert!(!selection.contains(".cloned()"));

        let preverified = source
            .split("pub async fn verified_authorization_proof_bytes(")
            .nth(1)
            .expect("preverified method")
            .split("// ---------------------------------------------------------------------------\n// Helper: all cheap admission checks")
            .next()
            .expect("preverified boundary");
        assert!(!preverified.contains("intent_bytes.clone"));
        assert!(!preverified.contains("Vec<Vec<u8>>"));
    }

    #[test]
    fn admission_accepts_tip_reward_for_the_next_block() {
        use jetsam_chain::consensus::params::coinbase_creation_id;

        let owner = Address([0xA5; 32]);
        let mint_height = 3;
        let mut state = ChainState::with_log_slots(6);
        state
            .state
            .set_slot(
                7,
                SlotValue::with_owner_fields(
                    1_000_000,
                    coinbase_creation_id(mint_height),
                    owner.as_fields(),
                ),
            )
            .unwrap();
        let genesis = genesis_header();
        let mut tip = genesis.clone();
        tip.height = mint_height;
        let mut headers = HashMap::new();
        headers.insert(0, genesis);
        headers.insert(mint_height, tip);
        let view = ChainView::new(mint_height, headers, 1, state.state);
        let mempool_state = |view: ChainView| MempoolState {
            pool: jetsam_chain::Mempool::new(16),
            view,
            floor: crate::floor::FeeFloor::new(4),
            admitted_input_slots: HashSet::new(),
            admitted_output_slots: HashSet::new(),
            client_rules: MempoolConfig::default().client_object_rules,
            vault_rules: MempoolConfig::default().vault_rules,
            destination: Default::default(),
            guardian: Default::default(),
            guardian_rules: MempoolConfig::default().guardian_rules,
        };

        let spend = |fee: u64| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            inputs[0] = TxInput {
                slot_index: 7,
                amount: 1_000_000,
                creation_id: coinbase_creation_id(mint_height),
            };
            let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
            outputs[0] = TxOutput {
                slot_index: 8,
                amount: 1_000_000 - fee,
                owner: Address([0xB6; 32]),
            };
            vec![TxPage::new(TxBody {
                epoch_anchor: [1; 32],
                fee,
                input_owner: owner,
                inputs,
                outputs,
                validity_bitmap: 1
                    | output_bitmap_bit(0)
                    | PAGED_SPEND_START_BIT
                    | PAGED_SPEND_END_BIT,
                is_coinbase: false,
            })
            .unwrap()]
        };

        // Learn the consensus fee, then anchor the child-height-4 candidate
        // to the epoch id captured by the accepted height-3 view.
        let probe_state = mempool_state(view);
        let required = jetsam_chain::consensus::fee_breakdown(
            1,
            1,
            probe_state.view.active_slot_count,
            probe_state.view.log_slots(),
        )
        .required_total;
        let mut candidate = spend(required);
        candidate[0].body.epoch_anchor = probe_state.view.user_epoch_anchor_id;
        let facts = validate_paged_spend(&candidate).unwrap();
        run_admission_checks(&candidate, &facts, &probe_state)
            .expect("accepted tip reward is spendable in its child block");
    }

    // --- M-1 (audit 2026-11-10): what the pool admits and the floor it sets ---

    /// What every M-1 input slot holds: 1 000 JTM.
    const M1_INPUT: u64 = 1_000_000_000;

    /// A view whose slots `1..=inputs` each hold [`M1_INPUT`] for the
    /// returned owner (minted at height 3), at `log_slots`, and a pool over it
    /// whose authorization check is a stand-in: these tests are about fees
    /// and block resources, and the proof is checked after both.
    fn m1_pool(inputs: u32, log_slots: usize) -> (AsyncMempool, ChainView, Address) {
        use jetsam_chain::consensus::params::coinbase_creation_id;

        let owner = Address([0xA5; 32]);
        let mut state = ChainState::with_log_slots(log_slots);
        // One root computation for every input (a debug build pays seconds
        // for each).
        let slots: Vec<(u32, SlotValue)> = (1..=inputs)
            .map(|slot| {
                let value = SlotValue::with_owner_fields(
                    M1_INPUT,
                    coinbase_creation_id(3),
                    owner.as_fields(),
                );
                (slot, value)
            })
            .collect();
        if !slots.is_empty() {
            state.state.apply_delta(&slots).unwrap();
        }
        let genesis = genesis_header();
        let mut tip = genesis.clone();
        tip.height = 3;
        let mut headers = HashMap::new();
        headers.insert(0, genesis);
        headers.insert(3, tip);
        let view = ChainView::new(3, headers, u64::from(inputs), state.state);
        let pool = AsyncMempool::new(view.clone(), MempoolConfig::default())
            .with_authorization_verification_executor(std::sync::Arc::new(|_task| Ok(())));
        (pool, view, owner)
    }

    /// The intent spending input slot `input` of [`m1_pool`] to `outputs`
    /// (slot, amount) pairs with `fee`, paged as the consensus requires, and
    /// its wire bytes.
    fn m1_intent(
        view: &ChainView,
        owner: Address,
        input: u32,
        outputs: &[(u32, u64)],
        fee: u64,
    ) -> (jetsam_tx::PagedSpendIntent, Vec<u8>) {
        use jetsam_chain::consensus::params::coinbase_creation_id;

        let page_count = outputs.len().div_ceil(TX_OUTPUTS).max(1);
        let pages = (0..page_count)
            .map(|index| {
                let mut inputs = [TxInput::dummy(); TX_INPUTS];
                let mut bitmap = 0u16;
                if index == 0 {
                    inputs[0] = TxInput {
                        slot_index: input,
                        amount: M1_INPUT,
                        creation_id: coinbase_creation_id(3),
                    };
                    bitmap |= 1 | PAGED_SPEND_START_BIT;
                }
                if index + 1 == page_count {
                    bitmap |= PAGED_SPEND_END_BIT;
                }
                let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
                for (offset, (slot, amount)) in outputs
                    .iter()
                    .skip(index * TX_OUTPUTS)
                    .take(TX_OUTPUTS)
                    .enumerate()
                {
                    page_outputs[offset] = TxOutput {
                        slot_index: *slot,
                        amount: *amount,
                        owner: Address([0xB6; 32]),
                    };
                    bitmap |= output_bitmap_bit(offset);
                }
                TxPage::new(TxBody {
                    epoch_anchor: view.user_epoch_anchor_id,
                    fee: if index == 0 { fee } else { 0 },
                    input_owner: owner,
                    inputs,
                    outputs: page_outputs,
                    validity_bitmap: bitmap,
                    is_coinbase: false,
                })
                .unwrap()
            })
            .collect();
        let intent = jetsam_tx::PagedSpendIntent::new(pages, vec![0xA5; 64]).unwrap();
        let bytes = intent.to_bytes().unwrap();
        (intent, bytes)
    }

    /// A spend of input slot `input` to one output (slot `input + 100`)
    /// with `fee`, or to two when `two_outputs`.
    fn m1_simple(
        view: &ChainView,
        owner: Address,
        input: u32,
        fee: u64,
        two_outputs: bool,
    ) -> (jetsam_tx::PagedSpendIntent, Vec<u8>) {
        let left = M1_INPUT - fee;
        let outputs = if two_outputs {
            vec![(input + 100, left / 2), (input + 200, left - left / 2)]
        } else {
            vec![(input + 100, left)]
        };
        m1_intent(view, owner, input, &outputs, fee)
    }

    fn m1_consensus_fee(view: &ChainView, inputs: u64, outputs: u64) -> u64 {
        jetsam_chain::consensus::fee_breakdown(
            inputs,
            outputs,
            view.active_slot_count,
            view.log_slots(),
        )
        .required_total
    }

    /// M-1 (a): a fee the pool admitted but no block ever carried used to fix
    /// the floor every later admission paid — an unminable 10 JTM intent made
    /// every node refuse ordinary fees for as long as it stayed. Only what is
    /// mined says what the network pays.
    #[tokio::test]
    async fn a_fee_admitted_but_never_mined_sets_no_floor() {
        let intents = crate::floor::FEE_FLOOR_MIN_SAMPLES as u32;
        let (pool, view, owner) = m1_pool(intents + 1, 8);
        for input in 1..=intents {
            let (attacker, bytes) = m1_simple(&view, owner, input, 10_000_000, false);
            pool.submit(attacker, bytes)
                .await
                .expect("a 10 JTM fee is admitted");
        }

        let ordinary = m1_consensus_fee(&view, 1, 1);
        let (victim, bytes) = m1_simple(&view, owner, intents + 1, ordinary, false);
        pool.submit(victim, bytes)
            .await
            .expect("the consensus minimum is admitted beside an unmined 10 JTM intent");
    }

    /// M-1: the floor is 0.9 x the median fee RATE (per weight unit,
    /// `MempoolEntry::compute_fee_rate`) of the pool's transactions blocks
    /// carried, asked of each transaction for its own weight; under
    /// `FEE_FLOOR_MIN_SAMPLES` mined samples there is no floor.
    #[tokio::test]
    async fn the_floor_is_the_fee_rate_of_mined_transactions_for_each_shape() {
        use crate::floor::FEE_FLOOR_MIN_SAMPLES;
        use crate::SubmitError;
        use jetsam_chain::consensus::ConsensusError;

        let samples = FEE_FLOOR_MIN_SAMPLES as u32;
        let (pool, view, owner) = m1_pool(samples + 4, 8);
        // 1 JTM over one input and one output: weight 2, rate 500 000.
        let mined_fee = 1_000_000;
        let mut mined = Vec::new();
        for input in 1..=samples {
            let (intent, bytes) = m1_simple(&view, owner, input, mined_fee, false);
            mined.push(pool.submit(intent, bytes).await.expect("admitted"));
        }
        // One more stays in the pool: the floor is forgotten only when the
        // pool drains.
        let (waiting, bytes) = m1_simple(&view, owner, samples + 1, mined_fee, false);
        pool.submit(waiting, bytes).await.expect("admitted");

        // One sample short: no floor yet.
        pool.on_new_block(&mined[..mined.len() - 1], 4, view.clone())
            .await;
        let (below, bytes) = m1_simple(
            &view,
            owner,
            samples + 2,
            m1_consensus_fee(&view, 1, 1),
            false,
        );
        pool.submit(below, bytes)
            .await
            .expect("no floor below FEE_FLOOR_MIN_SAMPLES mined samples");

        pool.on_new_block(&mined[mined.len() - 1..], 5, view.clone())
            .await;
        // Rate floor 0.9 x 500 000 = 450 000; one input and two outputs weigh
        // 1 + 2 + 4 x 1 = 7: 3 150 000 asked, where the old absolute floor
        // asked 0.9 JTM of every shape.
        let required = 3_150_000;
        let (short, bytes) = m1_simple(&view, owner, samples + 3, required - 1, true);
        match pool.submit(short, bytes).await {
            Err(SubmitError::Consensus(ConsensusError::BelowMinFee {
                required: asked, ..
            })) => {
                assert_eq!(asked, required, "the rate floor times this shape's weight")
            }
            other => panic!("a fee under the mined rate floor must be refused, got {other:?}"),
        }
        let (enough, bytes) = m1_simple(&view, owner, samples + 3, required, true);
        pool.submit(enough, bytes)
            .await
            .expect("the mined rate floor for this shape is admitted");
    }

    /// M-1 (main): the template builder skips every group that, with the
    /// coinbase's reserved segment, touches more than 256 segments
    /// (`TemplateResourceSelection::with_group`), so such a group is never
    /// mined; the pool used to admit it, hold it and relay it. It is refused
    /// at admission now, by the template's own predicate (its bound is
    /// pinned in `jetsam_chain::consensus::template`), before any state read.
    #[tokio::test]
    async fn admission_refuses_a_group_no_template_can_carry() {
        use crate::SubmitError;

        // log_slots 24: 256 segments of 2^16 slots. The input is in segment 0,
        // one output in each of the 255 others: 256, plus the coinbase's. The
        // input need not be live: the group is refused before any state read.
        let (pool, view, owner) = m1_pool(0, 24);
        let fee = m1_consensus_fee(&view, 1, 255);
        let mut outputs: Vec<(u32, u64)> = (1..=255u32).map(|segment| (segment << 16, 1)).collect();
        outputs[254].1 = M1_INPUT - fee - 254;
        let (intent, bytes) = m1_intent(&view, owner, 1, &outputs, fee);
        match pool.submit(intent, bytes).await {
            Err(SubmitError::NoTemplateCarries(_)) => {}
            other => panic!("a group no template carries must be refused, got {other:?}"),
        }
        assert_eq!(pool.len().await, 0);
    }

    /// M3.8: a transaction paying a v1.5 client object travels with its
    /// object and is never admitted to the pool; the node checks it with
    /// every admission rule but the pool's own (marker policy, conflicts with
    /// pool entries) — and its authorization, which a forged one fails.
    #[tokio::test]
    async fn a_client_payment_is_checked_like_an_admission_but_never_admitted() {
        use crate::SubmitError;
        use jetsam_chain::consensus::params::coinbase_creation_id;

        let owner = Address([0xA5; 32]);
        let mint_height = 3;
        let mut state = ChainState::with_log_slots(6);
        state
            .state
            .set_slot(
                7,
                SlotValue::with_owner_fields(
                    1_000_000,
                    coinbase_creation_id(mint_height),
                    owner.as_fields(),
                ),
            )
            .unwrap();
        let genesis = genesis_header();
        let mut tip = genesis.clone();
        tip.height = mint_height;
        let mut headers = HashMap::new();
        headers.insert(0, genesis);
        headers.insert(mint_height, tip);
        let view = ChainView::new(mint_height, headers, 1, state.state);
        let anchor = view.user_epoch_anchor_id;
        let required =
            jetsam_chain::consensus::fee_breakdown(1, 1, view.active_slot_count, view.log_slots())
                .required_total;
        let pool = AsyncMempool::new(view, MempoolConfig::default());
        let payment = |slot: u32, epoch_anchor: [u8; 32]| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            inputs[0] = TxInput {
                slot_index: slot,
                amount: 1_000_000,
                creation_id: coinbase_creation_id(mint_height),
            };
            let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
            outputs[0] = TxOutput {
                slot_index: 8,
                amount: 1_000_000 - required,
                owner: Address([0xB6; 32]),
            };
            jetsam_tx::PagedSpendIntent::new(
                vec![TxPage::new(TxBody {
                    epoch_anchor,
                    fee: required,
                    input_owner: owner,
                    inputs,
                    outputs,
                    validity_bitmap: 1
                        | output_bitmap_bit(0)
                        | PAGED_SPEND_START_BIT
                        | PAGED_SPEND_END_BIT,
                    is_coinbase: false,
                })
                .unwrap()],
                vec![0xA5; 64],
            )
            .unwrap()
        };
        // Every state rule is checked...
        assert!(matches!(
            pool.check_client_payment(&payment(7, [9; 32])).await,
            Err(SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::BadEpochAnchor
            ))
        ));
        assert!(pool
            .check_client_payment(&payment(9, anchor))
            .await
            .is_err());
        // ... then the authorization, which this stand-in fails.
        assert!(matches!(
            pool.check_client_payment(&payment(7, anchor)).await,
            Err(SubmitError::InvalidProof(_))
        ));
        // Nothing was admitted.
        assert_eq!(pool.state.lock().await.pool.len(), 0);
    }

    /// Admission asks the view for the anchors the *next block's* generation
    /// accepts, and nothing else decides. With the v1.3 clock dormant that is
    /// the single current anchor at every height — a page bound to the older
    /// one is refused exactly as it always was — and the pair the same view
    /// hands out is the one that would open under v1.3. The clock is the only
    /// thing holding that door.
    #[test]
    fn admission_asks_the_next_blocks_generation_which_anchors_it_accepts() {
        use crate::SubmitError;
        use jetsam_chain::consensus::params::{coinbase_creation_id, HistoryStepPackGeneration};

        let owner = Address([0xA5; 32]);
        let mint_height = 3;
        let mut state = ChainState::with_log_slots(6);
        state
            .state
            .set_slot(
                7,
                SlotValue::with_owner_fields(
                    1_000_000,
                    coinbase_creation_id(mint_height),
                    owner.as_fields(),
                ),
            )
            .unwrap();
        let genesis = genesis_header();
        let mut tip = genesis.clone();
        tip.height = mint_height;
        let mut headers = HashMap::new();
        headers.insert(0, genesis);
        headers.insert(mint_height, tip);
        let mut view = ChainView::new(mint_height, headers, 1, state.state);
        let current = [0x5C; 32];
        let previous = [0x6D; 32];
        let stale = [0x7E; 32];
        view.user_epoch_anchor_id = current;
        view.previous_user_epoch_anchor_id = previous;

        let st = MempoolState {
            pool: jetsam_chain::Mempool::new(16),
            view,
            floor: crate::floor::FeeFloor::new(4),
            admitted_input_slots: HashSet::new(),
            admitted_output_slots: HashSet::new(),
            client_rules: MempoolConfig::default().client_object_rules,
            vault_rules: MempoolConfig::default().vault_rules,
            destination: Default::default(),
            guardian: Default::default(),
            guardian_rules: MempoolConfig::default().guardian_rules,
        };
        let required = jetsam_chain::consensus::fee_breakdown(
            1,
            1,
            st.view.active_slot_count,
            st.view.log_slots(),
        )
        .required_total;
        let spend_at = |anchor: [u8; 32]| {
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            inputs[0] = TxInput {
                slot_index: 7,
                amount: 1_000_000,
                creation_id: coinbase_creation_id(mint_height),
            };
            let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
            outputs[0] = TxOutput {
                slot_index: 8,
                amount: 1_000_000 - required,
                owner: Address([0xB6; 32]),
            };
            vec![TxPage::new(TxBody {
                epoch_anchor: anchor,
                fee: required,
                input_owner: owner,
                inputs,
                outputs,
                validity_bitmap: 1
                    | output_bitmap_bit(0)
                    | PAGED_SPEND_START_BIT
                    | PAGED_SPEND_END_BIT,
                is_coinbase: false,
            })
            .unwrap()]
        };
        let verdict = |anchor: [u8; 32]| {
            let pages = spend_at(anchor);
            let facts = validate_paged_spend(&pages).unwrap();
            run_admission_checks(&pages, &facts, &st).map(|_| ())
        };

        verdict(current).expect("the current anchor is accepted by every generation");
        assert!(
            matches!(
                verdict(previous),
                Err(SubmitError::Consensus(
                    jetsam_chain::consensus::ConsensusError::BadEpochAnchor
                ))
            ),
            "with the v1.3 clock dormant the older anchor is refused, as it always was"
        );
        assert!(matches!(
            verdict(stale),
            Err(SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::BadEpochAnchor
            ))
        ));

        // The discriminant: same view, same pair — only the generation moves.
        let (accepted, generation) = st.view.accepted_user_epoch_anchors();
        assert_eq!(
            generation,
            HistoryStepPackGeneration::at_height(mint_height + 1),
            "admission judges by the generation of the block being built"
        );
        assert_eq!(generation, HistoryStepPackGeneration::V1);
        assert!(accepted.accepts(current, HistoryStepPackGeneration::V1));
        assert!(!accepted.accepts(previous, HistoryStepPackGeneration::V1));
        assert!(accepted.accepts(previous, HistoryStepPackGeneration::V1_3));
        assert!(!accepted.accepts(stale, HistoryStepPackGeneration::V1_3));
    }

    #[test]
    fn input_state_match_binds_creation_id() {
        let owner = Address([0xA5; 32]);
        let mut state = ChainState::with_log_slots(6);
        state
            .state
            .set_slot(
                7,
                SlotValue::with_owner_fields(1_000, 42, owner.as_fields()),
            )
            .unwrap();
        let mut headers = HashMap::new();
        headers.insert(0, genesis_header());
        let view = ChainView::new(0, headers, 1, state.state);
        let mut inputs = [TxInput::dummy(); TX_INPUTS];
        inputs[0] = TxInput {
            slot_index: 7,
            amount: 1_000,
            creation_id: 41,
        };
        let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
        outputs[0] = TxOutput {
            slot_index: 8,
            amount: 999,
            owner: Address([0xB6; 32]),
        };
        let mut pages = vec![TxPage::new(TxBody {
            epoch_anchor: [1; 32],
            fee: 1,
            input_owner: owner,
            inputs,
            outputs,
            validity_bitmap: 1 | output_bitmap_bit(0) | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT,
            is_coinbase: false,
        })
        .unwrap()];
        let mut facts = validate_paged_spend(&pages).unwrap();

        assert!(check_input_slots(&pages, &facts, &view).is_err());
        pages[0].body.inputs[0].creation_id = 42;
        facts = validate_paged_spend(&pages).unwrap();
        assert!(check_input_slots(&pages, &facts, &view).is_ok());
    }

    /// The v1.5 height of the crossing tests: the public profile's empty
    /// catalogue, armed at a height of their own (the release arms it at J).
    const CROSSING_J: u64 = 1_200;

    fn armed_rules() -> jetsam_chain::consensus::client_objects::ClientObjectRules {
        jetsam_chain::consensus::client_objects::ClientObjectRules {
            activation_height: Some(CROSSING_J),
            catalogue: &[],
            ..jetsam_chain::consensus::client_objects::ClientObjectRules::CONSENSUS
        }
    }

    /// The view of a chain whose tip is `tip`, every header around J known
    /// (both epoch anchors of any generation).
    fn view_of(state: &ChainState, tip: u64) -> ChainView {
        let genesis = genesis_header();
        let mut headers = HashMap::new();
        headers.insert(0, genesis.clone());
        for height in (CROSSING_J - 200)..=tip {
            let mut header = genesis.clone();
            header.height = height;
            header.nonce = u128::from(height);
            headers.insert(height, header);
        }
        ChainView::new(tip, headers, 1, state.state.clone())
    }

    /// A pool on an armed clock, its view at `tip`, and one ordinary
    /// transaction paying an owner that carries the registration marker tag
    /// (what `walletBuildClientPayment` builds), admissible for the block at
    /// `tip + 1` while that block is below J.
    struct MarkerCrossing {
        state: ChainState,
        intent: jetsam_tx::PagedSpendIntent,
    }

    impl MarkerCrossing {
        fn new(tip: u64) -> Self {
            use jetsam_chain::consensus::params::coinbase_creation_id;
            let owner = Address([0xA5; 32]);
            let mut state = ChainState::with_log_slots(6);
            state
                .state
                .set_slot(
                    7,
                    SlotValue::with_owner_fields(
                        1_000_000,
                        coinbase_creation_id(3),
                        owner.as_fields(),
                    ),
                )
                .unwrap();
            let view = view_of(&state, tip);
            let required = jetsam_chain::consensus::fee_breakdown(
                1,
                1,
                view.active_slot_count,
                view.log_slots(),
            )
            .required_total
                * 4;
            let mut marker = [0x5Au8; 32];
            marker[..8].copy_from_slice(
                &jetsam_chain::consensus::client_objects::CLIENT_REGISTRATION_MARKER_TAG,
            );
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            inputs[0] = TxInput {
                slot_index: 7,
                amount: 1_000_000,
                creation_id: coinbase_creation_id(3),
            };
            let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
            outputs[0] = TxOutput {
                slot_index: 8,
                amount: 1_000_000 - required,
                owner: Address(marker),
            };
            let pages = vec![TxPage::new(TxBody {
                epoch_anchor: view.user_epoch_anchor_id,
                fee: required,
                input_owner: owner,
                inputs,
                outputs,
                validity_bitmap: 1
                    | output_bitmap_bit(0)
                    | PAGED_SPEND_START_BIT
                    | PAGED_SPEND_END_BIT,
                is_coinbase: false,
            })
            .unwrap()];
            Self {
                state,
                intent: jetsam_tx::PagedSpendIntent::new(pages, vec![0xA5; 64]).unwrap(),
            }
        }

        fn view_at(&self, tip: u64) -> ChainView {
            view_of(&self.state, tip)
        }

        fn pool_at(&self, tip: u64) -> AsyncMempool {
            AsyncMempool::new(
                self.view_at(tip),
                MempoolConfig::default().with_client_object_rules(armed_rules()),
            )
            // This test is about the pool's v1.5 rule, not the proof.
            .with_authorization_verification_executor(std::sync::Arc::new(|_task| Ok(())))
        }

        async fn submit(&self, pool: &AsyncMempool) -> jetsam_poseidon2b::primitives::TxBodyHash {
            let bytes = self.intent.to_bytes().unwrap();
            pool.submit(self.intent.clone(), bytes)
                .await
                .expect("admitted while the next block is below J")
        }
    }

    /// B1 (review of v1.5.0): a transaction paying a marker owner is
    /// admitted while the next block is below J (the rule is inert there, as
    /// in v1.4.3). Once the tip reaches J - 1 it is a transaction no block at
    /// J can carry: the pool evicts it, and the template of J never sees it,
    /// so no miner proves and seals a block the chain refuses.
    #[tokio::test]
    async fn crossing_j_evicts_a_marker_payment_admitted_below_j() {
        use jetsam_chain::consensus::client_objects::{check_plain_transaction, ClientObjectError};
        let crossing = MarkerCrossing::new(CROSSING_J - 2);
        let pool = crossing.pool_at(CROSSING_J - 2);
        let mut events = pool.subscribe();
        let hash = crossing.submit(&pool).await;
        assert_eq!(pool.len().await, 1);

        // The tip reaches J - 1 (a block that did not include it).
        pool.on_new_block(&[], CROSSING_J - 1, crossing.view_at(CROSSING_J - 1))
            .await;
        assert_eq!(pool.len().await, 0, "a block at J can never carry it");
        let (anchors, generation) = pool.state.lock().await.view.accepted_user_epoch_anchors();
        assert!(pool
            .select_for_block_at_anchors(BLOCK_MAX_USER_PAGES, anchors, generation)
            .await
            .is_empty());
        let mut evicted = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let crate::event::MempoolEvent::TxEvicted { hash, reason } = event {
                evicted.push((hash, reason));
            }
        }
        assert_eq!(
            evicted,
            vec![(hash, crate::event::EvictReason::ClientObjectRule)],
            "the sender's wallet is told, and releases its coins"
        );
        // What the block at J would have been refused for.
        assert_eq!(
            check_plain_transaction(&crossing.intent.pages, CROSSING_J, &armed_rules()),
            Err(ClientObjectError::PaymentWithoutObject)
        );
        // And the pool refuses it again from now on.
        let bytes = crossing.intent.to_bytes().unwrap();
        assert!(pool.submit(crossing.intent.clone(), bytes).await.is_err());
    }

    /// Below J nothing changes (v1.4.3): the same transaction stays in the
    /// pool while the next block is below J, and is offered to it.
    #[tokio::test]
    async fn below_j_a_marker_payment_stays_a_plain_transaction() {
        let crossing = MarkerCrossing::new(CROSSING_J - 3);
        let pool = crossing.pool_at(CROSSING_J - 3);
        crossing.submit(&pool).await;
        pool.on_new_block(&[], CROSSING_J - 2, crossing.view_at(CROSSING_J - 2))
            .await;
        assert_eq!(pool.len().await, 1);
        assert_eq!(pool.evict_unminable_at(CROSSING_J - 1).await, 0);
        let (anchors, generation) = pool.state.lock().await.view.accepted_user_epoch_anchors();
        assert_eq!(
            pool.select_for_block_at_anchors(BLOCK_MAX_USER_PAGES, anchors, generation)
                .await
                .len(),
            1
        );
    }

    /// A view installed without a block (`update_chain_view`, a reorg or a
    /// snapshot) is judged the same way.
    #[tokio::test]
    async fn a_view_replaced_across_j_evicts_what_its_next_block_refuses() {
        let crossing = MarkerCrossing::new(CROSSING_J - 2);
        let pool = crossing.pool_at(CROSSING_J - 2);
        crossing.submit(&pool).await;
        pool.update_chain_view(crossing.view_at(CROSSING_J + 3))
            .await;
        assert_eq!(pool.len().await, 0);
    }

    // -----------------------------------------------------------------------
    // v1.7 (T2.5): the bridge vault's rules at admission and at J
    // -----------------------------------------------------------------------

    /// The vault rules of the crossing tests: J at `CROSSING_J`, the bridge
    /// opening `F` eight blocks later.
    fn armed_vault_rules() -> jetsam_chain::consensus::vault::VaultRules {
        jetsam_chain::consensus::vault::VaultRules {
            activation_height: Some(CROSSING_J),
            deposits_from_height: Some(CROSSING_J + 8),
            ..jetsam_chain::consensus::vault::VaultRules::current()
        }
    }

    /// One transaction spending the live slot 7 of `input_owner` into
    /// `outputs` (slots 8, 9), fee four times the consensus minimum.
    struct VaultCase {
        state: ChainState,
        intent: jetsam_tx::PagedSpendIntent,
    }

    impl VaultCase {
        /// Bound to the epoch anchor of the child of `tip`.
        fn new(tip: u64, input_owner: Address, outputs: &[(Address, u64)]) -> Self {
            use jetsam_chain::consensus::params::coinbase_creation_id;
            let mut state = ChainState::with_log_slots(6);
            let required = jetsam_chain::consensus::fee_breakdown(
                1,
                outputs.len() as u64,
                1,
                state.state.log_slots() as u32,
            )
            .required_total
                * 4;
            let input = outputs.iter().map(|(_, a)| a).sum::<u64>() + required;
            state
                .state
                .set_slot(
                    7,
                    SlotValue::with_owner_fields(
                        input,
                        coinbase_creation_id(3),
                        input_owner.as_fields(),
                    ),
                )
                .unwrap();
            let view = view_of(&state, tip);
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            inputs[0] = TxInput {
                slot_index: 7,
                amount: input,
                creation_id: coinbase_creation_id(3),
            };
            let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
            let mut bitmap = 1 | PAGED_SPEND_START_BIT | PAGED_SPEND_END_BIT;
            for (i, (owner, amount)) in outputs.iter().enumerate() {
                page_outputs[i] = TxOutput {
                    slot_index: 8 + i as u32,
                    amount: *amount,
                    owner: *owner,
                };
                bitmap |= output_bitmap_bit(i);
            }
            let pages = vec![TxPage::new(TxBody {
                epoch_anchor: view.user_epoch_anchor_id,
                fee: required,
                input_owner,
                inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
            .unwrap()];
            Self {
                state,
                intent: jetsam_tx::PagedSpendIntent::new(pages, vec![0xA5; 64]).unwrap(),
            }
        }

        /// The view at `tip`, with the vault's `bridge_locked` from J on.
        fn view_at(&self, tip: u64, bridge_locked: u64) -> ChainView {
            let mut view = view_of(&self.state, tip);
            if tip + 1 >= CROSSING_J {
                view.vault = Some(crate::view::VaultView {
                    bridge_locked,
                    released: Default::default(),
                    floor: 0,
                    committees: Vec::new(),
                    advances: Default::default(),
                });
            }
            view
        }

        fn pool_at(&self, tip: u64, bridge_locked: u64) -> AsyncMempool {
            AsyncMempool::new(
                self.view_at(tip, bridge_locked),
                MempoolConfig::default()
                    .with_client_object_rules(armed_rules())
                    .with_vault_rules(armed_vault_rules()),
            )
            .with_authorization_verification_executor(std::sync::Arc::new(|_task| Ok(())))
        }

        async fn submit(
            &self,
            pool: &AsyncMempool,
        ) -> Result<jetsam_poseidon2b::primitives::TxBodyHash, super::SubmitError> {
            let bytes = self.intent.to_bytes().unwrap();
            pool.submit(self.intent.clone(), bytes).await
        }
    }

    fn vault_error(
        result: Result<jetsam_poseidon2b::primitives::TxBodyHash, super::SubmitError>,
    ) -> Option<jetsam_chain::consensus::vault::VaultError> {
        use jetsam_chain::consensus::destination::state::DestinationBlockError;
        match result {
            Err(super::SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::DestinationBlock(
                    DestinationBlockError::Vault(error),
                ),
            )) => Some(error),
            _ => None,
        }
    }

    /// V1 at admission (SPEC retour §6.2, T2.5): from J on, a transaction
    /// spending from the vault never enters the pool (a release travels with
    /// its object); below J the vault address is an ordinary owner.
    #[tokio::test]
    async fn mempool_refuses_plain_vault_spend() {
        use jetsam_chain::consensus::vault::{VaultError, VAULT_ADDRESS};
        let case = VaultCase::new(
            CROSSING_J - 2,
            VAULT_ADDRESS,
            &[(Address([0x42; 32]), 5_000_000)],
        );
        let above = case.pool_at(CROSSING_J + 3, 0);
        assert_eq!(
            vault_error(case.submit(&above).await),
            Some(VaultError::SpendFromLockedAddress { group: 0 })
        );
        assert_eq!(above.len().await, 0);
        let below = case.pool_at(CROSSING_J - 2, 0);
        assert!(
            case.submit(&below).await.is_ok(),
            "below J: an ordinary owner"
        );
    }

    /// A vault spend admitted while the next block is below J is evicted
    /// when the tip reaches J − 1, with its own reason; the template of J
    /// never sees it.
    #[tokio::test]
    async fn eviction_at_j_crossing() {
        use crate::event::{EvictReason, MempoolEvent};
        use jetsam_chain::consensus::vault::VAULT_ADDRESS;
        let case = VaultCase::new(
            CROSSING_J - 2,
            VAULT_ADDRESS,
            &[(Address([0x42; 32]), 5_000_000)],
        );
        let pool = case.pool_at(CROSSING_J - 2, 0);
        let mut events = pool.subscribe();
        let hash = case.submit(&pool).await.expect("below J");
        pool.on_new_block(&[], CROSSING_J - 1, case.view_at(CROSSING_J - 1, 0))
            .await;
        assert_eq!(pool.len().await, 0, "a block at J can never carry it");
        let mut evicted = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let MempoolEvent::TxEvicted { hash, reason } = event {
                evicted.push((hash, reason));
            }
        }
        assert_eq!(evicted, vec![(hash, EvictReason::VaultRule)]);
        assert!(EvictReason::VaultRule
            .operator_explanation()
            .contains("vault"));
        // The same through a replaced view and a lagging template.
        let pool = case.pool_at(CROSSING_J - 2, 0);
        case.submit(&pool).await.expect("below J");
        pool.update_chain_view(case.view_at(CROSSING_J + 3, 0))
            .await;
        assert_eq!(pool.len().await, 0);
        let pool = case.pool_at(CROSSING_J - 2, 0);
        case.submit(&pool).await.expect("below J");
        assert_eq!(pool.evict_unminable_at(CROSSING_J).await, 1);
        assert_eq!(pool.vault_rules(), armed_vault_rules());
    }

    /// V8 at admission against the tip's `bridge_locked` (A1 M4): a deposit
    /// that would take the vault over its ceiling is refused; one that fits
    /// is admitted (above the bridge opening, V4-d).
    #[tokio::test]
    async fn mempool_refuses_deposit_over_vault_ceiling_at_admission() {
        use jetsam_chain::consensus::params::MICRO_PER_JTM;
        use jetsam_chain::consensus::vault::{
            deposit_marker, VaultError, VAULT_ADDRESS, VAULT_OUTSTANDING_CEIL_MICRO,
        };
        let tip = CROSSING_J + 20;
        let locked = VAULT_OUTSTANDING_CEIL_MICRO - 2 * MICRO_PER_JTM;
        let deposit = |amount| {
            VaultCase::new(
                tip,
                Address([0xA5; 32]),
                &[(VAULT_ADDRESS, amount), (deposit_marker(&[0x42; 20]), 0)],
            )
        };
        let over = deposit(3 * MICRO_PER_JTM);
        assert_eq!(
            vault_error(over.submit(&over.pool_at(tip, locked)).await),
            Some(VaultError::VaultCeilingReached {
                group: 0,
                locked,
                amount: 3 * MICRO_PER_JTM,
                ceil: VAULT_OUTSTANDING_CEIL_MICRO
            })
        );
        let fits = deposit(2 * MICRO_PER_JTM);
        assert!(fits.submit(&fits.pool_at(tip, locked)).await.is_ok());
        // At or below F (V4-d), refused whatever the room.
        assert_eq!(
            vault_error(fits.submit(&fits.pool_at(CROSSING_J + 7, 0)).await),
            Some(VaultError::DepositBeforeBridgeOpening {
                group: 0,
                height: CROSSING_J + 8,
                opening: CROSSING_J + 8
            })
        );
    }

    /// The floor of the release tests: `alloc_counter(J − 1)`.
    const RELEASE_FLOOR: u64 = 10;

    /// A vault slot created after the floor (not by a coinbase) and the
    /// release intent whose group spends it whole: `(owner, amount − R)` and
    /// the marker, fee `R`, bound to the epoch anchor of the child of `tip`;
    /// its objects are those of the synthetic signed L1 (A2-M1), which prove
    /// the burn `(4, [0xa1; 32], 5 000 000)` only. The view knows the anchor
    /// committee and holds its keys.
    struct ReleaseCase {
        state: ChainState,
        intent: crate::destination::DestinationIntent,
        n: u64,
    }

    impl ReleaseCase {
        fn new(tip: u64, n: u64, amount: u64, paid_delta: i64) -> Self {
            let l1 = crate::destination::tests::SyntheticL1::load();
            let release = jetsam_chain::consensus::vault::VaultRelease {
                n,
                owner: Address([0xa1; 32]),
                amount,
                storage_proof: l1.release.storage_proof.clone(),
            };
            let payee = release.owner;
            Self::paying(tip, release, payee, paid_delta)
        }

        /// RET3: the release over Base of the burn of the Base fixture
        /// (`(3, [0xa1; 32], 5 000 000)`), its group paying `payee`.
        fn over_base(tip: u64, payee: Address) -> Self {
            let base = crate::destination::tests::SyntheticBase::load();
            Self::over_base_of(tip, base.release, payee)
        }

        /// The release over Base of `release` under the Base fixture's
        /// proofs, its group paying `payee`.
        fn over_base_of(
            tip: u64,
            release: jetsam_chain::consensus::vault::VaultRelease,
            payee: Address,
        ) -> Self {
            let base = crate::destination::tests::SyntheticBase::load();
            let mut case = Self::paying(tip, release, payee, 0);
            let crate::destination::DestinationIntent::Release(release) = &mut case.intent else {
                unreachable!()
            };
            release.finality = base.finality.clone();
            release.account_proof = base.account_proof.clone();
            release.base = Some(base.proofs());
            case
        }

        fn paying(
            tip: u64,
            release: jetsam_chain::consensus::vault::VaultRelease,
            payee: Address,
            paid_delta: i64,
        ) -> Self {
            use jetsam_chain::consensus::vault::{RELEASE_FEE_MICRO, VAULT_ADDRESS};
            let l1 = crate::destination::tests::SyntheticL1::load();
            let rf = RELEASE_FEE_MICRO;
            let (n, amount) = (release.n, release.amount);
            let paid = (amount - rf).saturating_add_signed(paid_delta);
            let mut state = ChainState::with_log_slots(6);
            state
                .state
                .set_slot(
                    7,
                    SlotValue::with_owner_fields(paid + rf, 50, VAULT_ADDRESS.as_fields()),
                )
                .unwrap();
            let view = view_of(&state, tip);
            let mut inputs = [TxInput::dummy(); TX_INPUTS];
            inputs[0] = TxInput {
                slot_index: 7,
                amount: paid + rf,
                creation_id: 50,
            };
            let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
            outputs[0] = TxOutput {
                slot_index: 8,
                amount: paid,
                owner: payee,
            };
            outputs[1] = TxOutput {
                slot_index: 9,
                amount: 0,
                owner: release.marker(),
            };
            let pages = vec![TxPage::new(TxBody {
                epoch_anchor: view.user_epoch_anchor_id,
                fee: rf,
                input_owner: VAULT_ADDRESS,
                inputs,
                outputs,
                validity_bitmap: 1
                    | output_bitmap_bit(0)
                    | output_bitmap_bit(1)
                    | PAGED_SPEND_START_BIT
                    | PAGED_SPEND_END_BIT,
                is_coinbase: false,
            })
            .unwrap()];
            Self {
                state,
                intent: crate::destination::DestinationIntent::Release(
                    crate::destination::ReleaseIntent {
                        release,
                        finality: l1.finality.clone(),
                        account_proof: l1.account_proof.clone(),
                        payment: jetsam_tx::PagedSpendIntent::new(pages, vec![0xA5; 64]).unwrap(),
                        base: None,
                    },
                ),
                n,
            }
        }

        /// The same intent with `edit` applied to its release.
        fn with(&self, edit: impl FnOnce(&mut crate::destination::ReleaseIntent)) -> Self {
            let crate::destination::DestinationIntent::Release(mut release) = self.intent.clone()
            else {
                unreachable!()
            };
            edit(&mut release);
            Self {
                state: self.state.clone(),
                intent: crate::destination::DestinationIntent::Release(release),
                n: self.n,
            }
        }

        fn view_at(&self, tip: u64, paid: &[u64]) -> ChainView {
            let mut view = view_of(&self.state, tip);
            if tip + 1 >= CROSSING_J {
                let l1 = crate::destination::tests::SyntheticL1::load();
                let mut released = jetsam_chain::consensus::vault::ReleasedSet::new();
                for n in paid {
                    released.insert(*n);
                }
                view.vault = Some(crate::view::VaultView {
                    bridge_locked: 1_000_000_000,
                    released,
                    floor: RELEASE_FLOOR,
                    committees: vec![l1.anchor_committee()],
                    advances: Default::default(),
                });
                view.destination_rules = l1.rules(armed_vault_rules());
                view.committee_keys = std::sync::Arc::new(l1.anchor_keys());
            }
            view
        }

        fn pool_at(&self, tip: u64, paid: &[u64]) -> AsyncMempool {
            Self::pool_on(self.view_at(tip, paid))
        }

        fn pool_on(view: ChainView) -> AsyncMempool {
            AsyncMempool::new(
                view,
                MempoolConfig::default()
                    .with_client_object_rules(armed_rules())
                    .with_vault_rules(armed_vault_rules()),
            )
            .with_authorization_verification_executor(std::sync::Arc::new(|_task| Ok(())))
        }

        async fn submit(
            &self,
            pool: &AsyncMempool,
        ) -> Result<crate::destination::DestinationAdmitted, super::SubmitError> {
            pool.submit_destination(&self.intent.encode().unwrap())
                .await
        }
    }

    /// The destination-block refusal of a destination submission.
    fn destination_refusal(
        result: Result<crate::destination::DestinationAdmitted, super::SubmitError>,
    ) -> Option<jetsam_chain::consensus::destination::state::DestinationBlockError> {
        match result {
            Err(super::SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::DestinationBlock(error),
            )) => Some(error),
            _ => None,
        }
    }

    /// A2-M1: with the keys of the committees its branch knows, the pool
    /// verifies a release as the block does — the finality update (BLS),
    /// the bridge account under its finalized root, the storage proof (V7) —
    /// and refuses, unheld, what the block would refuse: a storage proof
    /// that proves nothing, an index never burned, another amount than the
    /// burn, a finality update signed by a committee the branch does not
    /// know, an account proof that does not open the root, a genesis proof
    /// that does not open the contract's `genesisId` (A5-8). Without the keys
    /// (or before any committee) it refuses softly: the node's own fault.
    #[tokio::test]
    async fn bogus_release_intent_is_refused_at_admission_when_keys_are_held() {
        use jetsam_chain::consensus::destination::light_client::LightClientError;
        use jetsam_chain::consensus::destination::state::DestinationBlockError as E;
        use jetsam_chain::consensus::vault::{
            decode_bridge_account_proof, encode_bridge_account_proof, VaultError,
        };
        let tip = CROSSING_J + 20;
        let l1 = crate::destination::tests::SyntheticL1::load();
        let case = ReleaseCase::new(tip, 4, 5_000_000, 0);
        let (real_account, real_genesis) = decode_bridge_account_proof(&l1.account_proof).unwrap();
        let account = encode_bridge_account_proof(&[vec![0xdd; 64]], &real_genesis);
        // A5-8: the genesis proof replaced by the withdrawal's storage path.
        let no_genesis = encode_bridge_account_proof(&real_account, &l1.release.storage_proof);
        let bogus: [(ReleaseCase, fn(&E) -> bool); 6] = [
            (
                case.with(|r| r.release.storage_proof = vec![vec![0xee; 40]]),
                |e| matches!(e, E::Vault(VaultError::NotProvenOnL1 { n: 4, .. })),
            ),
            (ReleaseCase::new(tip, 5, 5_000_000, 0), |e| {
                matches!(e, E::Vault(VaultError::NotProvenOnL1 { n: 5, .. }))
            }),
            (ReleaseCase::new(tip, 4, 6_000_000, 0), |e| {
                matches!(e, E::Vault(VaultError::NotProvenOnL1 { n: 4, .. }))
            }),
            (case.with(|r| r.finality = l1.finality_stranger.clone()), |e| {
                *e == E::LightClient(LightClientError::BadSignature)
            }),
            (case.with(|r| r.account_proof = account.clone()), |e| {
                matches!(e, E::BridgeAccount(_))
            }),
            (case.with(|r| r.account_proof = no_genesis.clone()), |e| {
                matches!(
                    e,
                    E::BridgeAccount(
                        jetsam_chain::consensus::destination::l1_read::L1ReadError::Mpt(_)
                    )
                )
            }),
        ];
        for (i, (bogus, expected)) in bogus.iter().enumerate() {
            // Each on the state its group spends.
            let pool = bogus.pool_at(tip, &[]);
            let refusal = destination_refusal(bogus.submit(&pool).await);
            assert!(
                refusal.as_ref().is_some_and(expected),
                "bogus intent {i}: {refusal:?}"
            );
            assert!(pool.destination_candidates().await.1.is_empty(), "not held");
        }
        let pool = case.pool_at(tip, &[]);
        case.submit(&pool).await.expect("the burn's own release");
        assert_eq!(pool.destination_candidates().await.1.len(), 1);

        // This node lacks the keys, or its branch knows no committee yet:
        // refused, softly, and not held.
        let mut keyless = case.view_at(tip, &[]);
        keyless.committee_keys = std::sync::Arc::default();
        let mut blind = case.view_at(tip, &[]);
        blind.vault.as_mut().unwrap().committees.clear();
        for (view, expected) in [
            (
                keyless,
                E::CommitteeKeysUnavailable {
                    root: l1.anchor.committee_root,
                },
            ),
            (blind, E::NoCommitteeKnown),
        ] {
            let pool = ReleaseCase::pool_on(view);
            let result = case.submit(&pool).await;
            assert!(
                result.as_ref().is_err_and(super::SubmitError::is_soft),
                "{result:?}"
            );
            assert_eq!(destination_refusal(result), Some(expected));
            assert!(pool.destination_candidates().await.1.is_empty());
        }
    }

    /// A2-M1: what the L1 verification costs one admission (printed; run
    /// with `--ignored`): the release (finality BLS, account, V7) and the
    /// committee update signed by the known committee (BLS), alone; one whole
    /// release admission (authorization stubbed); the copy of the key cache
    /// a view takes per block.
    #[tokio::test]
    #[ignore]
    async fn measure_destination_admission_cost() {
        use crate::destination::verify_destination_intent;
        use std::time::Instant;
        let tip = CROSSING_J + 20;
        let l1 = crate::destination::tests::SyntheticL1::load();
        let case = ReleaseCase::new(tip, 4, 5_000_000, 0);
        let rules = l1.rules(armed_vault_rules());
        let keys = l1.anchor_keys();
        let known = [l1.anchor_committee()];
        let update = crate::destination::DestinationIntent::CommitteeUpdate(
            crate::destination::CommitteeUpdateIntent {
                payload: l1.next_update.clone(),
                payment: case.intent.payment().clone(),
            },
        );
        let runs = 40u32;
        let time = |f: &dyn Fn()| {
            f();
            let start = Instant::now();
            for _ in 0..runs {
                f();
            }
            start.elapsed() / runs
        };
        let release = time(&|| {
            verify_destination_intent(&case.intent, &known, &rules, &keys).unwrap();
        });
        let committee = time(&|| {
            verify_destination_intent(&update, &known, &rules, &keys).unwrap();
        });
        let copy = time(&|| {
            std::hint::black_box(keys.clone());
        });
        let decompress = time(&|| {
            let mut cache = jetsam_chain::consensus::destination::state::CommitteeKeyCache::default();
            cache.insert_update_payload(&l1.anchor_update).unwrap();
        });
        let pool = case.pool_at(tip, &[]);
        let bytes = case.intent.encode().unwrap();
        pool.submit_destination(&bytes).await.unwrap();
        let start = Instant::now();
        for _ in 0..runs {
            pool.submit_destination(&bytes).await.unwrap();
        }
        let admission = start.elapsed() / runs;
        println!(
            "A2-M1 cost per call ({runs} runs): release verification {release:?}, \
             committee-update verification {committee:?}, whole release admission \
             {admission:?}, key cache copy (one committee) {copy:?}, decompression of one \
             committee's keys (the block's, not the admission's) {decompress:?}"
        );
    }

    /// A2-M1: a release verified outside the lock is judged again under it
    /// when the pool's view moved to other committees meanwhile (a new tip
    /// arrived during its authorization proof): one the new committees no
    /// longer verify is refused and not held; the same release with the
    /// view unmoved is held.
    #[tokio::test]
    async fn a_release_is_judged_again_when_the_view_moves_to_other_committees() {
        use jetsam_chain::consensus::destination::state::{CommitteeRef, DestinationBlockError as E};
        let tip = CROSSING_J + 20;
        let l1 = crate::destination::tests::SyntheticL1::load();
        let case = ReleaseCase::new(tip, 4, 5_000_000, 0);
        let later = CommitteeRef {
            period: l1.anchor.period + 2,
            root: [0x77; 32],
        };
        let mut moved = case.view_at(tip, &[]);
        moved.vault.as_mut().unwrap().committees = vec![later];
        let pool = ReleaseCase::pool_on(case.view_at(tip, &[]));
        let shared = pool.clone();
        let next_view = std::sync::Mutex::new(Some(moved));
        let pool = pool.with_authorization_verification_executor(std::sync::Arc::new(
            move |_task| {
                if let Some(view) = next_view.lock().unwrap().take() {
                    shared.state.blocking_lock().view = view;
                }
                Ok(())
            },
        ));
        assert_eq!(
            destination_refusal(case.submit(&pool).await),
            Some(E::CommitteeKeysUnavailable { root: later.root }),
            "judged by the committees the view moved to"
        );
        assert!(pool.destination_candidates().await.1.is_empty());
        case.submit(&case.pool_at(tip, &[]))
            .await
            .expect("unmoved, the same release is held");
    }

    /// A2-M1: a release the pool holds, verified, is never replaced by an
    /// intent of the same index that does not verify (a forged storage
    /// proof, a finality update of another committee): the honest one stays
    /// held, as it was.
    #[tokio::test]
    async fn a_held_verified_release_is_not_replaced_by_an_unverified_one() {
        let tip = CROSSING_J + 20;
        let l1 = crate::destination::tests::SyntheticL1::load();
        let case = ReleaseCase::new(tip, 4, 5_000_000, 0);
        let pool = case.pool_at(tip, &[]);
        case.submit(&pool).await.expect("the honest release");
        let crate::destination::DestinationIntent::Release(honest) = &case.intent else {
            unreachable!()
        };
        for forged in [
            case.with(|r| r.release.storage_proof = vec![vec![0xee; 40]]),
            case.with(|r| r.finality = l1.finality_stranger.clone()),
        ] {
            assert!(forged.submit(&pool).await.is_err());
            assert_eq!(
                pool.destination_candidates().await.1,
                vec![honest.clone()],
                "the honest release stays held"
            );
        }
    }

    /// A2-M1: a committee update is verified at admission as the block
    /// verifies it: while the branch knows no committee, only the anchor
    /// update (the one whose next committee is the anchor's) is held; once it
    /// knows one, only an update that committee signs, whose next committee
    /// its branch proves. A refused update replaces nothing; without the
    /// keys of the known committee the refusal is soft.
    #[tokio::test]
    async fn bogus_committee_update_is_refused_at_admission_when_keys_are_held() {
        use jetsam_chain::consensus::destination::light_client::LightClientError;
        use jetsam_chain::consensus::destination::ssz::LightClientUpdate;
        use jetsam_chain::consensus::destination::state::{
            committee_update_marker, DestinationBlockError as E,
        };
        let tip = CROSSING_J + 20;
        let l1 = crate::destination::tests::SyntheticL1::load();
        let intent = |payload: &[u8]| {
            let case = VaultCase::new(
                tip,
                Address([0xA5; 32]),
                &[
                    (Address([0x42; 32]), 10),
                    (committee_update_marker(payload), 0),
                ],
            );
            let intent = crate::destination::DestinationIntent::CommitteeUpdate(
                crate::destination::CommitteeUpdateIntent {
                    payload: payload.to_vec(),
                    payment: case.intent.clone(),
                },
            );
            (case, intent.encode().unwrap())
        };
        let view = |case: &VaultCase, known: bool, keys: bool| {
            let mut view = case.view_at(tip, 0);
            view.destination_rules = l1.rules(armed_vault_rules());
            if known {
                view.vault.as_mut().unwrap().committees = vec![l1.anchor_committee()];
            }
            if keys {
                view.committee_keys = std::sync::Arc::new(l1.anchor_keys());
            }
            view
        };
        let pool_on = |view: ChainView| {
            AsyncMempool::new(
                view,
                MempoolConfig::default()
                    .with_client_object_rules(armed_rules())
                    .with_vault_rules(armed_vault_rules()),
            )
            .with_authorization_verification_executor(std::sync::Arc::new(|_task| Ok(())))
        };
        let tampered = {
            let mut update = LightClientUpdate::decode_payload(&l1.next_update).unwrap();
            update.next_sync_committee.pubkeys[0][0] ^= 1;
            update.encode_payload()
        };
        let forged_signature = {
            let mut update = LightClientUpdate::decode_payload(&l1.next_update).unwrap();
            update.sync_aggregate.bits[0] ^= 1;
            update.encode_payload()
        };

        // The branch knows no committee: the anchor update only.
        let (anchor_case, anchor_bytes) = intent(&l1.anchor_update);
        let pool = pool_on(view(&anchor_case, false, false));
        for payload in [&l1.next_update, &tampered] {
            let (_, bytes) = intent(payload);
            assert_eq!(
                destination_refusal(pool.submit_destination(&bytes).await),
                Some(E::AnchorMismatch)
            );
        }
        assert!(pool.destination_candidates().await.0.is_empty());
        assert_eq!(
            pool.submit_destination(&anchor_bytes).await.unwrap(),
            crate::destination::DestinationAdmitted::CommitteeUpdate {
                period: l1.anchor.period
            }
        );

        // It knows the anchor committee and holds its keys: the update it
        // signs is held; a tampered one is refused and replaces nothing.
        let (next_case, next_bytes) = intent(&l1.next_update);
        let pool = pool_on(view(&next_case, true, true));
        assert_eq!(
            pool.submit_destination(&next_bytes).await.unwrap(),
            crate::destination::DestinationAdmitted::CommitteeUpdate {
                period: l1.anchor.period + 1
            }
        );
        for (payload, expected) in [
            (&tampered, E::LightClient(LightClientError::NextCommitteeBranch)),
            (&forged_signature, E::LightClient(LightClientError::BadSignature)),
            (&l1.anchor_update, E::LightClient(LightClientError::PeriodMismatch {
                known: l1.anchor.period,
                update: l1.anchor.period - 1,
            })),
        ] {
            let (_, bytes) = intent(payload);
            assert_eq!(
                destination_refusal(pool.submit_destination(&bytes).await),
                Some(expected)
            );
            let held = pool.destination_candidates().await.0;
            assert_eq!(held.len(), 1);
            assert_eq!(held[0].payload, l1.next_update, "the signed update stays held");
        }

        // Without the keys of the committee it knows: soft, not held.
        let pool = pool_on(view(&next_case, true, false));
        let result = pool.submit_destination(&next_bytes).await;
        assert!(result.as_ref().is_err_and(super::SubmitError::is_soft), "{result:?}");
        assert_eq!(
            destination_refusal(result),
            Some(E::CommitteeKeysUnavailable {
                root: l1.anchor.committee_root
            })
        );
        assert!(pool.destination_candidates().await.0.is_empty());
    }

    /// The exact bytes of every `BridgeIntentAdmitted` event received so far.
    fn relayed_bridge_intents(
        events: &mut tokio::sync::broadcast::Receiver<crate::event::MempoolEvent>,
    ) -> Vec<Vec<u8>> {
        let mut relayed = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let crate::event::MempoolEvent::BridgeIntentAdmitted { intent_bytes } = event {
                relayed.push(intent_bytes.to_vec());
            }
        }
        relayed
    }

    /// T7.1: a held destination intent is announced once, with its exact
    /// bytes, for the node to relay it to its peers; a refused one never is.
    #[tokio::test]
    async fn admitted_destination_intent_is_announced_with_its_bytes() {
        let tip = CROSSING_J + 20;
        let case = ReleaseCase::new(tip, 4, 5_000_000, 0);
        let pool = case.pool_at(tip, &[]);
        let mut events = pool.subscribe();
        case.submit(&pool).await.expect("a valid release intent");
        assert_eq!(
            relayed_bridge_intents(&mut events),
            vec![case.intent.encode().unwrap()]
        );
        // Already paid by the branch: refused, nothing to relay.
        let paid = case.pool_at(tip, &[4]);
        let mut paid_events = paid.subscribe();
        assert!(case.submit(&paid).await.is_err());
        assert!(relayed_bridge_intents(&mut paid_events).is_empty());
    }

    /// T2.5: a release intent (object, finality update, account proof, and
    /// the release group spending the vault) is held for the template; the
    /// same group alone is a plain vault spend and is refused; an index the
    /// branch has paid, a group that pays the wrong amount, and anything
    /// below J are refused.
    #[tokio::test]
    async fn mempool_admits_release_intent_with_valid_objects() {
        use crate::event::MempoolEvent;
        use jetsam_chain::consensus::vault::VaultError;
        let tip = CROSSING_J + 20;
        let case = ReleaseCase::new(tip, 4, 5_000_000, 0);
        let pool = case.pool_at(tip, &[]);
        // A release's fee is capped at R by the vault gate (D59): the pool's
        // dynamic floor, raised far above it here, never refuses one.
        {
            let mut locked = pool.state.lock().await;
            for _ in 0..64 {
                locked.floor.record_mined(10_000_000);
            }
            assert!(
                locked.floor.current().required(1, 1)
                    > jetsam_chain::consensus::vault::RELEASE_FEE_MICRO
            );
        }
        let mut events = pool.subscribe();
        let admitted = case.submit(&pool).await.expect("a valid release intent");
        assert_eq!(
            admitted,
            crate::destination::DestinationAdmitted::Release { n: 4 }
        );
        let (updates, releases) = pool.destination_candidates().await;
        assert!(updates.is_empty());
        assert_eq!(releases.len(), 1);
        assert_eq!(releases[0].release.n, case.n);
        assert_eq!(pool.len().await, 0, "never in the plain pool");
        let mut seen = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let MempoolEvent::DestinationObjectAdmitted { release } = event {
                seen.push(release);
            }
        }
        assert_eq!(seen, vec![Some(4)], "the miner is woken");
        // The group alone, as a plain transaction (on a pool at its usual
        // floor: the raised one would refuse its fee first).
        let payment = case.intent.payment().clone();
        let bytes = payment.to_bytes().unwrap();
        assert_eq!(
            vault_error(case.pool_at(tip, &[]).submit(payment, bytes).await),
            Some(VaultError::SpendFromLockedAddress { group: 0 })
        );
        // Already paid by the branch.
        let paid = case.pool_at(tip, &[4]);
        assert!(matches!(
            case.submit(&paid).await,
            Err(super::SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::DestinationBlock(
                    jetsam_chain::consensus::destination::state::DestinationBlockError::Vault(
                        VaultError::ReleaseAlreadyPaid { n: 4 }
                    )
                )
            ))
        ));
        // A group paying one µJTM more than `amount − R`.
        let greedy = ReleaseCase::new(tip, 4, 5_000_000, 1);
        assert!(matches!(
            greedy.submit(&greedy.pool_at(tip, &[])).await,
            Err(super::SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::DestinationBlock(
                    jetsam_chain::consensus::destination::state::DestinationBlockError::Vault(
                        VaultError::ReleasePaysWrongAmount { .. }
                    )
                )
            ))
        ));
        // Below J.
        let early = ReleaseCase::new(CROSSING_J - 5, 4, 5_000_000, 0);
        assert!(matches!(
            early.submit(&early.pool_at(CROSSING_J - 5, &[])).await,
            Err(super::SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::Destination(
                    jetsam_chain::consensus::destination::DestinationError::SectionBeforeActivation
                )
            ))
        ));
    }

    /// Held releases are forgotten once their index is paid on the branch or
    /// their vault slots are spent, at a new block or a replaced view.
    #[tokio::test]
    async fn destination_intents_are_pruned_when_paid_or_spent() {
        let tip = CROSSING_J + 20;
        let case = ReleaseCase::new(tip, 4, 5_000_000, 0);
        let pool = case.pool_at(tip, &[]);
        case.submit(&pool).await.expect("admitted");
        pool.on_new_block(&[], tip + 1, case.view_at(tip + 1, &[]))
            .await;
        assert_eq!(
            pool.destination_candidates().await.1.len(),
            1,
            "still payable"
        );
        pool.on_new_block(&[], tip + 2, case.view_at(tip + 2, &[4]))
            .await;
        assert!(pool.destination_candidates().await.1.is_empty(), "paid");

        let pool = case.pool_at(tip, &[]);
        case.submit(&pool).await.expect("admitted");
        let mut spent = ChainState::with_log_slots(6);
        spent.state.set_slot(7, SlotValue::EMPTY).unwrap();
        let mut view = view_of(&spent, tip + 1);
        view.vault = case.view_at(tip + 1, &[]).vault;
        pool.update_chain_view(view).await;
        assert!(pool.destination_candidates().await.1.is_empty(), "spent");
    }

    /// A committee update with the transaction that pays for it (its
    /// `JTMLCU01` marker) is held; the same transaction alone is refused (a
    /// marker without its object); the update is forgotten once the branch
    /// knows its period.
    #[tokio::test]
    async fn mempool_holds_committee_update_with_its_payment() {
        use jetsam_chain::consensus::destination::state::{committee_update_marker, CommitteeRef};
        use jetsam_chain::consensus::vault::VaultError;
        let tip = CROSSING_J + 20;
        let payload = crate::destination::tests::committee_update_payload();
        let marker = committee_update_marker(&payload);
        let case = VaultCase::new(
            tip,
            Address([0xA5; 32]),
            &[(Address([0x42; 32]), 10), (marker, 0)],
        );
        // A2-M1: the real update is the anchor update of a branch whose
        // anchor is its next committee (the pool verifies it as the block).
        let next = jetsam_chain::consensus::destination::ssz::LightClientUpdate::decode_payload(
            &payload,
        )
        .unwrap()
        .next_sync_committee
        .hash_tree_root();
        let mut view = case.view_at(tip, 0);
        view.destination_rules = jetsam_chain::consensus::destination::state::DestinationRules {
            activation_height: Some(CROSSING_J),
            anchor: Some(jetsam_chain::consensus::destination::state::L1Anchor {
                period: 1_895,
                committee_root: next,
            }),
            vault: armed_vault_rules(),
            ..jetsam_chain::consensus::destination::state::DestinationRules::CONSENSUS
        };
        let pool = ReleaseCase::pool_on(view);
        let intent = crate::destination::DestinationIntent::CommitteeUpdate(
            crate::destination::CommitteeUpdateIntent {
                payload,
                payment: case.intent.clone(),
            },
        );
        let crate::destination::DestinationIntent::CommitteeUpdate(update) = &intent else {
            unreachable!()
        };
        let period = update.next_period().unwrap();
        assert_eq!(
            pool.submit_destination(&intent.encode().unwrap())
                .await
                .expect("a committee update with its payment"),
            crate::destination::DestinationAdmitted::CommitteeUpdate { period }
        );
        assert_eq!(pool.destination_candidates().await.0.len(), 1);
        assert_eq!(
            vault_error(case.submit(&pool).await),
            Some(VaultError::ReleaseMarkerOutsideRelease { group: 0 })
        );
        // Below J, refused.
        let early = case.pool_at(CROSSING_J - 5, 0);
        assert!(matches!(
            early.submit_destination(&intent.encode().unwrap()).await,
            Err(super::SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::Destination(
                    jetsam_chain::consensus::destination::DestinationError::SectionBeforeActivation
                )
            ))
        ));
        let mut view = case.view_at(tip + 1, 0);
        view.vault.as_mut().unwrap().committees = vec![CommitteeRef {
            period,
            root: [0u8; 32],
        }];
        pool.update_chain_view(view).await;
        assert!(pool.destination_candidates().await.0.is_empty());
    }

    /// A2-M2: a transaction that pays for an object other than a release —
    /// a committee update's, a v1.5 client object's — is held to the vault
    /// rules of a plain transaction (its own committee-update marker aside):
    /// an output to the vault, a marker carrying value, a spend from the
    /// vault are refused at admission (no block could carry them: V1, V4),
    /// the committee update is never held. (The template half of A2-M2 is
    /// the test of the same name in `jetsam_miner::destination_slot`.)
    #[tokio::test]
    async fn object_payment_paying_the_vault_is_refused_at_admission_and_blamed_at_the_template() {
        use jetsam_chain::consensus::destination::state::{
            committee_update_marker, DestinationBlockError as E, DestinationRules, L1Anchor,
        };
        use jetsam_chain::consensus::vault::{VaultError, VAULT_ADDRESS};
        let tip = CROSSING_J + 20;
        let payload = crate::destination::tests::committee_update_payload();
        let marker = committee_update_marker(&payload);
        let anchor =
            jetsam_chain::consensus::destination::ssz::LightClientUpdate::decode_payload(&payload)
                .unwrap()
                .next_sync_committee
                .hash_tree_root();
        let update = |outputs: &[(Address, u64)], owner: Address| {
            let case = VaultCase::new(tip, owner, outputs);
            let mut view = case.view_at(tip, 0);
            view.destination_rules = DestinationRules {
                activation_height: Some(CROSSING_J),
                anchor: Some(L1Anchor {
                    period: 1_895,
                    committee_root: anchor,
                }),
                vault: armed_vault_rules(),
                ..DestinationRules::CONSENSUS
            };
            let intent = crate::destination::DestinationIntent::CommitteeUpdate(
                crate::destination::CommitteeUpdateIntent {
                    payload: payload.clone(),
                    payment: case.intent.clone(),
                },
            );
            (ReleaseCase::pool_on(view), intent.encode().unwrap())
        };
        for (outputs, owner, expected) in [
            (
                vec![(VAULT_ADDRESS, 10), (marker, 0)],
                Address([0xA5; 32]),
                VaultError::VaultOutputOutsideDeposit { group: 0 },
            ),
            (
                vec![(Address([0x42; 32]), 10), (marker, 1)],
                Address([0xA5; 32]),
                VaultError::MarkerCarriesValue { group: 0 },
            ),
            (
                vec![(Address([0x42; 32]), 10), (marker, 0)],
                VAULT_ADDRESS,
                VaultError::SpendFromLockedAddress { group: 0 },
            ),
        ] {
            let (pool, bytes) = update(&outputs, owner);
            assert_eq!(
                destination_refusal(pool.submit_destination(&bytes).await),
                Some(E::Vault(expected))
            );
            assert!(pool.destination_candidates().await.0.is_empty());
        }
        let (pool, bytes) = update(
            &[(Address([0x42; 32]), 10), (marker, 0)],
            Address([0xA5; 32]),
        );
        pool.submit_destination(&bytes)
            .await
            .expect("the same update, its payment paying no vault");

        // A v1.5 client object's payment, checked like an admission.
        let client = VaultCase::new(tip, Address([0xA5; 32]), &[(VAULT_ADDRESS, 10)]);
        let pool = client.pool_at(tip, 0);
        assert!(matches!(
            pool.check_client_payment(&client.intent).await,
            Err(super::SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::DestinationBlock(E::Vault(
                    VaultError::VaultOutputOutsideDeposit { group: 0 }
                ))
            ))
        ));
        let plain = VaultCase::new(tip, Address([0xA5; 32]), &[(Address([0x42; 32]), 10)]);
        plain
            .pool_at(tip, 0)
            .check_client_payment(&plain.intent)
            .await
            .expect("the same payment, paying no vault");
    }

    /// A template for J built while the pool's view still lags at J - 2 (its
    /// parent J - 1 committed, the pool not told yet): the builder asks the
    /// pool to drop what the child cannot carry before it selects.
    #[tokio::test]
    async fn a_lagging_view_is_judged_for_the_templates_own_child() {
        let crossing = MarkerCrossing::new(CROSSING_J - 2);
        let pool = crossing.pool_at(CROSSING_J - 2);
        crossing.submit(&pool).await;
        assert_eq!(pool.evict_unminable_at(CROSSING_J).await, 1);
        assert_eq!(pool.len().await, 0);
        assert_eq!(pool.client_object_rules(), armed_rules());
    }

    // -----------------------------------------------------------------------
    // v1.7, the return over Base (D101, D107, D114): dormant on every profile
    // -----------------------------------------------------------------------

    /// The return-over-Base height of these tests, above the crossing's J.
    const BASE_HEIGHT: u64 = CROSSING_J + 10;

    /// `view` with the return over Base armed at [`BASE_HEIGHT`] and the
    /// branch holding `advances`.
    fn over_base_view(
        mut view: ChainView,
        advances: &[(u64, jetsam_chain::consensus::destination::advance::AdvanceEntry)],
    ) -> ChainView {
        let l1 = crate::destination::tests::SyntheticL1::load();
        view.destination_rules =
            crate::destination::tests::SyntheticBase::rules(&l1, armed_vault_rules(), BASE_HEIGHT);
        if let Some(vault) = view.vault.as_mut() {
            for (n, entry) in advances {
                vault.advances.insert(*n, *entry);
            }
        }
        view
    }

    fn pool_over(view: ChainView) -> AsyncMempool {
        AsyncMempool::new(
            view,
            MempoolConfig::default()
                .with_client_object_rules(armed_rules())
                .with_vault_rules(armed_vault_rules()),
        )
        .with_authorization_verification_executor(std::sync::Arc::new(|_task| Ok(())))
    }

    const ALICE: Address = Address([0xa1; 32]);
    const PAYER: Address = Address([0x51; 32]);
    const BOB: Address = Address([0xb0; 32]);
    const STRANGER: Address = Address([0x5e; 32]);

    fn advance_refusal(
        result: Result<jetsam_poseidon2b::primitives::TxBodyHash, super::SubmitError>,
    ) -> Option<jetsam_chain::consensus::destination::advance::AdvanceError> {
        use jetsam_chain::consensus::destination::state::DestinationBlockError;
        match result {
            Err(super::SubmitError::Consensus(
                jetsam_chain::consensus::ConsensusError::DestinationBlock(
                    DestinationBlockError::Advance(error),
                ),
            )) => Some(error),
            _ => None,
        }
    }

    /// D107, D114: from the return-over-Base height on, the pool judges an
    /// advance or a transfer of its claim as the block judges its group,
    /// against the branch's advances: a malformed or underpaid one, a burn
    /// already advanced or paid, a transfer by another than the creditor are
    /// refused; an advance of another burn of `n` than one held is admitted
    /// (it blocks nothing). Below the height, and while the height is `None`,
    /// an advance-shaped group is an ordinary spend, as for a node without
    /// the rule.
    #[tokio::test]
    async fn over_base_the_pool_judges_an_advance_as_the_block_does() {
        use jetsam_chain::consensus::destination::advance::{
            advance_marker, AdvanceEntry, AdvanceError, AdvanceRules,
        };
        use jetsam_chain::consensus::vault::RELEASE_FEE_MICRO;
        let jtm = jetsam_chain::consensus::params::MICRO_PER_JTM;
        let amount = 10 * jtm;
        let paid = AdvanceRules::CONSENSUS.min_paid(amount, RELEASE_FEE_MICRO);
        let marker = advance_marker(3, &ALICE, amount, &PAYER);
        let tip = BASE_HEIGHT;
        let good = VaultCase::new(tip, PAYER, &[(ALICE, paid), (marker, 0)]);
        let with_value = VaultCase::new(tip, PAYER, &[(ALICE, paid), (marker, 9)]);
        let underpaid = VaultCase::new(tip, PAYER, &[(ALICE, paid - 1), (marker, 0)]);
        let held = AdvanceEntry {
            origin: BOB,
            creditor: BOB,
            owner: ALICE,
            amount,
        };
        let stranger = AdvanceEntry {
            origin: STRANGER,
            creditor: STRANGER,
            owner: STRANGER,
            amount: RELEASE_FEE_MICRO,
        };
        let pool_with = |case: &VaultCase, held: &[(u64, AdvanceEntry)], paid: &[u64]| {
            let mut view = over_base_view(case.view_at(tip, 0), held);
            for n in paid {
                view.vault.as_mut().unwrap().released.insert(*n);
            }
            pool_over(view)
        };
        // Admitted, alone or beside a stranger's advance of another burn of 3.
        for others in [vec![], vec![(3, stranger)]] {
            let pool = pool_with(&good, &others, &[]);
            good.submit(&pool).await.expect("an advance");
            assert_eq!(pool.len().await, 1);
        }
        for (case, held, paid, expected) in [
            (&with_value, vec![], vec![], AdvanceError::MarkerCarriesValue { group: 0 }),
            (
                &underpaid,
                vec![],
                vec![],
                AdvanceError::AdvanceUnderpaid {
                    group: 0,
                    paid: paid - 1,
                    min: paid,
                },
            ),
            (&good, vec![(3, held)], vec![], AdvanceError::AlreadyAdvanced { n: 3 }),
            (&good, vec![], vec![3], AdvanceError::AdvanceOfReleased { n: 3 }),
        ] {
            let pool = pool_with(case, &held, &paid);
            assert_eq!(advance_refusal(case.submit(&pool).await), Some(expected));
            assert_eq!(pool.len().await, 0);
        }
        // The claim's transfer: by its creditor only.
        let transfer = |from| VaultCase::new(tip, from, &[(PAYER, 0), (held.transfer_marker(3, &PAYER), 0)]);
        let by_creditor = transfer(BOB);
        let pool = pool_with(&by_creditor, &[(3, held)], &[]);
        by_creditor.submit(&pool).await.expect("the creditor transfers");
        let by_stranger = transfer(STRANGER);
        let pool = pool_with(&by_stranger, &[(3, held)], &[]);
        assert_eq!(
            advance_refusal(by_stranger.submit(&pool).await),
            Some(AdvanceError::TransferNotByCreditor { group: 0, n: 3 })
        );
        // Below the height, or dormant (this build): an ordinary spend.
        let below = VaultCase::new(BASE_HEIGHT - 2, PAYER, &[(ALICE, paid), (marker, 9)]);
        let pool = pool_over(over_base_view(below.view_at(BASE_HEIGHT - 2, 0), &[]));
        below.submit(&pool).await.expect("an ordinary spend below the height");
        let pool = with_value.pool_at(tip, 0);
        with_value.submit(&pool).await.expect("an ordinary spend while dormant");
    }

    /// An advance-shaped group admitted below the return-over-Base height is
    /// evicted when the next block reaches it if no block can carry it (a
    /// marker with value), with its own reason; one over its advancer's cap
    /// stays (it fits once an advance is consumed; the template leaves it out
    /// meanwhile).
    #[tokio::test]
    async fn crossing_the_return_over_base_evicts_the_advances_no_block_carries() {
        use crate::event::{EvictReason, MempoolEvent};
        use jetsam_chain::consensus::destination::advance::{
            advance_marker, AdvanceEntry, AdvanceRules,
        };
        use jetsam_chain::consensus::vault::RELEASE_FEE_MICRO;
        let jtm = jetsam_chain::consensus::params::MICRO_PER_JTM;
        let amount = 10 * jtm;
        let paid = AdvanceRules::CONSENSUS.min_paid(amount, RELEASE_FEE_MICRO);
        let marker = advance_marker(3, &ALICE, amount, &PAYER);
        let before = BASE_HEIGHT - 2;
        let bad = VaultCase::new(before, PAYER, &[(ALICE, paid), (marker, 9)]);
        let pool = pool_over(over_base_view(bad.view_at(before, 0), &[]));
        let mut events = pool.subscribe();
        let hash = bad.submit(&pool).await.expect("below the height");
        pool.on_new_block(&[], BASE_HEIGHT - 1, over_base_view(bad.view_at(BASE_HEIGHT - 1, 0), &[]))
            .await;
        assert_eq!(pool.len().await, 0, "no block from the height carries it");
        let mut evicted = Vec::new();
        while let Ok(event) = events.try_recv() {
            if let MempoolEvent::TxEvicted { hash, reason } = event {
                evicted.push((hash, reason));
            }
        }
        assert_eq!(evicted, vec![(hash, EvictReason::AdvanceRule)]);
        // Over the cap: kept.
        let full: Vec<(u64, AdvanceEntry)> = (10..13)
            .map(|n| {
                (
                    n,
                    AdvanceEntry {
                        origin: PAYER,
                        creditor: PAYER,
                        owner: ALICE,
                        amount: 1_000 * jtm,
                    },
                )
            })
            .collect();
        let capped = VaultCase::new(before, PAYER, &[(ALICE, paid), (marker, 0)]);
        let pool = pool_over(over_base_view(capped.view_at(before, 0), &full));
        capped.submit(&pool).await.expect("below the height");
        pool.on_new_block(&[], BASE_HEIGHT - 1, over_base_view(capped.view_at(BASE_HEIGHT - 1, 0), &full))
            .await;
        assert_eq!(pool.len().await, 1, "over the cap for now: kept");
        // Dormant: nothing evicted.
        let pool = bad.pool_at(before, 0);
        bad.submit(&pool).await.expect("dormant");
        pool.on_new_block(&[], BASE_HEIGHT - 1, bad.view_at(BASE_HEIGHT - 1, 0)).await;
        assert_eq!(pool.len().await, 1);
    }

    /// D101, D107, D114: from the return-over-Base height on, the pool admits
    /// a release proven over Base (game proof, output root preimage, bridge
    /// under the Base root, all verified as the block does) paying what V2
    /// asks of the branch: the creditor of the advance of exactly its burn,
    /// else the owner (an advance of another burn of `n` changes nothing).
    /// Refused: one paying another payee; a release read on the L1 after
    /// the height; one over Base below it; a game that does not hold; a burn
    /// owned by an advance marker. While the height is `None` (this build),
    /// its kind does not even decode. A held release paying the owner is
    /// forgotten once an advance of exactly its burn appears.
    #[tokio::test]
    async fn over_base_a_release_is_proven_over_base_and_pays_v2s_payee() {
        use jetsam_chain::consensus::destination::advance::{advance_marker, AdvanceEntry};
        use jetsam_chain::consensus::destination::state::DestinationBlockError as E;
        use jetsam_chain::consensus::vault::VaultError;
        let tip = BASE_HEIGHT + 5;
        let base = crate::destination::tests::SyntheticBase::load();
        let burn = base.release.clone();
        assert_eq!(burn.owner, ALICE);
        let advance = AdvanceEntry {
            origin: PAYER,
            creditor: BOB,
            owner: burn.owner,
            amount: burn.amount,
        };
        let stranger = AdvanceEntry {
            origin: STRANGER,
            creditor: STRANGER,
            owner: STRANGER,
            amount: jetsam_chain::consensus::vault::RELEASE_FEE_MICRO,
        };
        let to_owner = ReleaseCase::over_base(tip, ALICE);
        let to_creditor = ReleaseCase::over_base(tip, BOB);
        let pool_of = |case: &ReleaseCase, held: &[(u64, AdvanceEntry)], at: u64| {
            pool_over(over_base_view(case.view_at(at, &[]), held))
        };
        for held in [vec![], vec![(burn.n, stranger)]] {
            let pool = pool_of(&to_owner, &held, tip);
            assert_eq!(
                to_owner.submit(&pool).await.expect("admitted"),
                crate::destination::DestinationAdmitted::ReleaseOverBase { n: burn.n }
            );
            assert_eq!(pool.destination_candidates().await.1.len(), 1);
        }
        let pool = pool_of(&to_creditor, &[(burn.n, advance)], tip);
        to_creditor.submit(&pool).await.expect("repays the creditor");
        let refused = [
            (
                pool_of(&to_owner, &[(burn.n, advance)], tip),
                &to_owner,
                E::Vault(VaultError::ReleaseShape { group: 0 }),
            ),
            (
                pool_of(&to_creditor, &[], tip),
                &to_creditor,
                E::Vault(VaultError::ReleaseShape { group: 0 }),
            ),
        ];
        for (pool, case, expected) in refused {
            assert_eq!(destination_refusal(case.submit(&pool).await), Some(expected));
            assert!(pool.destination_candidates().await.1.is_empty());
        }
        // Read on the L1 after the height; over Base below it.
        let l1 = ReleaseCase::new(tip, 4, 5_000_000, 0);
        let pool = pool_over(over_base_view(l1.view_at(tip, &[]), &[]));
        assert_eq!(destination_refusal(l1.submit(&pool).await), Some(E::BaseProofMissing));
        let early = ReleaseCase::over_base(BASE_HEIGHT - 3, ALICE);
        let pool = pool_of(&early, &[], BASE_HEIGHT - 3);
        assert_eq!(
            destination_refusal(early.submit(&pool).await),
            Some(E::BaseObjectBeforeActivation)
        );
        // A game that does not hold: the preimage of another output root.
        let wrong = to_owner.with(|r| r.base.as_mut().unwrap().output_preimage[100] ^= 1);
        let pool = pool_of(&wrong, &[], tip);
        assert!(matches!(
            destination_refusal(wrong.submit(&pool).await),
            Some(E::BaseReturn(_))
        ));
        // A burn owned by an advance marker is never payable over Base.
        let tagged = advance_marker(9, &ALICE, 1, &PAYER);
        let marker_owner = ReleaseCase::over_base_of(
            tip,
            jetsam_chain::consensus::vault::VaultRelease {
                owner: tagged,
                ..burn.clone()
            },
            tagged,
        );
        let pool = pool_of(&marker_owner, &[], tip);
        assert_eq!(
            destination_refusal(marker_owner.submit(&pool).await),
            Some(E::Vault(VaultError::ReleaseShape { group: 0 }))
        );
        // Dormant (this build): kind 3 is unknown.
        let pool = to_owner.pool_at(tip, &[]);
        assert!(matches!(
            to_owner.submit(&pool).await,
            Err(super::SubmitError::MalformedIntent(_))
        ));
        // Held paying the owner, then an advance of exactly the burn: the
        // release no longer pays what V2 asks and is forgotten.
        let pool = pool_of(&to_owner, &[], tip);
        to_owner.submit(&pool).await.expect("held");
        pool.update_chain_view(over_base_view(to_owner.view_at(tip, &[]), &[(burn.n, advance)]))
            .await;
        assert!(pool.destination_candidates().await.1.is_empty());
    }
}
