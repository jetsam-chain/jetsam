// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! The aggregator's side of the v1.6 batch slot, for the
//! `jetsam-aggregate` tool and the miner's pre-pass: freeze a batch class,
//! build a batch of client proofs as a junction on the chain parent's lanes,
//! encode it, and decide a received one natively.
//!
//! A batch of `n` clients is built as a chain of steps: the first client in
//! a leaf, each next one folded into the previous batch, then the
//! **junction** — the top batch over the parent block's lanes (its incoming
//! lanes are the parent's). A batch of one client is its own junction (one
//! step). The batch is valid for the parent whose lanes it was built on, and
//! for every later parent that carried no other batch (a block without a
//! batch carries its parent's lanes).
//!
//! The batch class is an input here (its matrix frozen by
//! [`AggregationClass::freeze`]); the chain's is a consensus constant.

use std::collections::BTreeMap;
use std::sync::Arc;

use jetsam_ivc_core::field_r1cs::FieldR1cs;
use jetsam_ivc_core::matrix_claim::c1::C1MatrixAccClaim;
use jetsam_ivc_core::pcs::Commitment;

use super::chain_arm::prepare_chain_batch;
use super::step::{
    assemble_client_batch_step, decide_client_batch, freeze_client_batch_class,
    prepare_client_batch_step, prove_client_batch_step, ClientBatchClass, ClientBatchFirst,
    ClientBatchProof, ClientBatchStepInput,
};
use super::wire::{client_batch_proof_max_wire_bytes, decode_client_batch_proof, encode_client_batch_proof};
use crate::acceptance::history_step::{
    decode_history_step_client_proof, decode_history_step_client_zk_proof,
    HistoryStepClientRegistry, HistoryStepClientWitness, HistoryStepClientZkWitness,
    PreparedHistoryStepClient,
};
use crate::acceptance::history_step_bank::HistoryStepClientForm;

type Hash = [u8; 32];

/// A frozen batch class: the relation every step proves and every batch
/// slot verifies.
pub struct AggregationClass {
    form: HistoryStepClientForm,
    pub(crate) class: ClientBatchClass,
    /// The class matrix with its digest, as a v1.6 tip's `D_batch` lane is
    /// decided on it (built once, from the digest computed at the freeze).
    batch_matrix: std::sync::OnceLock<crate::acceptance::history_step_bank::HistoryStepBatchMatrix>,
}

impl AggregationClass {
    /// Freeze the batch class over the client `form`, at the smallest batch
    /// shape from `m_guess` up (production: the canonical form, m = 22).
    pub fn freeze(form: &HistoryStepClientForm, m_guess: usize) -> Result<Self, String> {
        let class = freeze_client_batch_class(form, m_guess).map_err(|error| format!("{error:?}"))?;
        Ok(Self {
            form: form.clone(),
            class,
            batch_matrix: std::sync::OnceLock::new(),
        })
    }

    /// The chain's batch class (v1.6): the class over the
    /// hiding client form at the production batch shape, frozen once per
    /// process (tens of seconds) and checked against the consensus constant
    /// `CLIENT_BATCH_CLASS`. Refused when the build names no batch class or
    /// derives another one: a node never decides a batch against a class the
    /// chain does not name.
    pub fn chain() -> Result<&'static AggregationClass, String> {
        use crate::acceptance::history_step_bank::HISTORY_STEP_BATCH_FORM_M;
        use jetsam_chain::consensus::client_objects::CLIENT_BATCH_CLASS;
        static CHAIN: std::sync::OnceLock<Result<AggregationClass, String>> = std::sync::OnceLock::new();
        CHAIN
            .get_or_init(|| {
                let class = Self::freeze(&HistoryStepClientForm::canonical_z(), HISTORY_STEP_BATCH_FORM_M)?;
                match CLIENT_BATCH_CLASS {
                    Some(expected) if expected == class.digest() && class.m() == HISTORY_STEP_BATCH_FORM_M => Ok(class),
                    Some(_) => Err("this build derives a batch class other than CLIENT_BATCH_CLASS".into()),
                    None => Err("this build names no batch class".into()),
                }
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    #[cfg(test)]
    pub(crate) fn of(form: &HistoryStepClientForm, class: ClientBatchClass) -> Self {
        Self {
            form: form.clone(),
            class,
            batch_matrix: std::sync::OnceLock::new(),
        }
    }

    /// `D_batch`: the structural digest of the class matrix.
    pub fn digest(&self) -> Hash {
        self.class.digest
    }

    /// The post-commit class digest.
    pub fn post_commit(&self) -> Hash {
        self.class.post_commit
    }

    /// The batch shape `m`.
    pub fn m(&self) -> usize {
        self.class.parts.shape.m
    }

    /// The class matrix's useful rows.
    pub fn useful_rows(&self) -> usize {
        self.class.matrix.useful_rows
    }

    /// The class matrix (the `D_batch` lane is evaluated on it).
    pub fn matrix(&self) -> Arc<FieldR1cs> {
        Arc::clone(&self.class.matrix)
    }

    /// Upper bound of a batch proof of this class on the wire.
    pub fn max_wire_bytes(&self) -> Result<usize, String> {
        client_batch_proof_max_wire_bytes(&self.class.parts).map_err(|error| format!("{error:?}"))
    }

    /// The client form whose proofs it aggregates.
    pub fn form(&self) -> &HistoryStepClientForm {
        &self.form
    }

    /// The class matrix and its digest `D_batch` (computed at the freeze, from
    /// that matrix), as a v1.6 tip's `D_batch` lane is decided on it.
    pub fn batch_matrix(&self) -> &crate::acceptance::history_step_bank::HistoryStepBatchMatrix {
        self.batch_matrix.get_or_init(|| {
            crate::acceptance::history_step_bank::HistoryStepBatchMatrix::of_frozen_class(
                self.class.digest,
                Arc::clone(&self.class.matrix),
            )
        })
    }
}

/// One client proof to aggregate: its encoding
/// (`encode_history_step_client_proof`, or `encode_history_step_client_zk_proof`
/// for a class over a hiding form) and the matrix registered at its `D`.
pub struct AggregationClient {
    pub proof_bytes: Vec<u8>,
    pub matrix: Arc<FieldR1cs>,
}

/// The lanes a junction is built on, and a batch decided against: the chain
/// parent's entry lanes and `D_batch` lane (dead at the first block after
/// v1.6, or after blocks that carried no batch since).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParentLanes {
    pub entries: Vec<Option<C1MatrixAccClaim>>,
    pub batch: Option<C1MatrixAccClaim>,
}

impl ParentLanes {
    /// Dead lanes over a registry of `capacity` entries.
    pub fn empty(capacity: usize) -> Self {
        Self {
            entries: vec![None; capacity],
            batch: None,
        }
    }

    /// The canonical encoding an aggregator reads over RPC: the
    /// entry count (one byte), then each entry lane and the `D_batch` lane,
    /// each a flag (0 dead, 1 live) followed, when live, by its point and
    /// value (32 bytes per F256 coordinate, little-endian).
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = vec![u8::try_from(self.entries.len()).expect("a registry of at most 255 entries")];
        for lane in self.entries.iter().chain(std::iter::once(&self.batch)) {
            match lane {
                None => bytes.push(0),
                Some(claim) => {
                    bytes.push(1);
                    for coordinate in claim.point.iter().chain(std::iter::once(&claim.value)) {
                        bytes.extend_from_slice(&coordinate.to_le_bytes());
                    }
                }
            }
        }
        bytes
    }

    /// Decode [`Self::encode`]'s bytes for a registry of `capacity` entries
    /// and lanes of a class with `k_log` (points of `2 k_log + 1`
    /// coordinates). Refused unless canonical and exact.
    pub fn decode(bytes: &[u8], capacity: usize, k_log: usize) -> Result<Self, String> {
        use jetsam_ivc_core::field::F256;
        let width = 2 * k_log + 1;
        let (&count, mut rest) = bytes.split_first().ok_or("empty parent lanes")?;
        if usize::from(count) != capacity {
            return Err(format!("parent lanes of {count} entries, the registry has {capacity}"));
        }
        let mut lane = || -> Result<Option<C1MatrixAccClaim>, String> {
            let (&flag, tail) = rest.split_first().ok_or("truncated parent lanes")?;
            rest = tail;
            match flag {
                0 => Ok(None),
                1 => {
                    let bytes_needed = 32 * (width + 1);
                    if rest.len() < bytes_needed {
                        return Err("truncated parent lane".into());
                    }
                    let (lane, tail) = rest.split_at(bytes_needed);
                    rest = tail;
                    let mut coordinates = lane
                        .chunks_exact(32)
                        .map(|chunk| F256::from_le_bytes(chunk.try_into().expect("32-byte chunk")))
                        .collect::<Vec<_>>();
                    let value = coordinates.pop().expect("a lane has a value");
                    Ok(Some(C1MatrixAccClaim {
                        point: coordinates,
                        value,
                    }))
                }
                other => Err(format!("parent lane flag {other}")),
            }
        };
        let entries = (0..capacity).map(|_| lane()).collect::<Result<Vec<_>, _>>()?;
        let batch = lane()?;
        if !rest.is_empty() {
            return Err("trailing bytes after the parent lanes".into());
        }
        let lanes = Self { entries, batch };
        if lanes.encode() != bytes {
            return Err("non-canonical parent lanes".into());
        }
        Ok(lanes)
    }
}

/// The lanes a v1.6 terminal publishes, as the parent lanes of a batch built
/// or decided over the block: its entry lanes and its `D_batch`
/// lane, read with a receiving node's canonicality rules. The terminal is
/// decoded, not verified (the caller holds it as its chain's tip). Refused
/// under a runtime without the batch slot.
pub fn terminal_parent_lanes(
    runtime: &crate::acceptance::history_step::HistoryStepRuntime,
    terminal_bytes: &[u8],
) -> Result<ParentLanes, String> {
    let lanes = runtime
        .bank()
        .layout()
        .batch
        .ok_or("a terminal of a relation without the batch slot carries no batch lanes")?;
    let terminal = crate::acceptance::history_step::decode_history_step_terminal(runtime, terminal_bytes)
        .map_err(|error| format!("terminal: {error:?}"))?;
    let claim = crate::acceptance::history_step_bank::parse_history_step_batch_lanes(&lanes, terminal.proof().io())
        .map_err(|error| format!("batch lanes: {error:?}"))?;
    Ok(ParentLanes {
        entries: claim.entries,
        batch: claim.batch_lane,
    })
}

/// What a batch covers, as its payment names it and its block publishes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchStatement {
    /// Root of the ordered `(D, io_commitment)` tree.
    pub statements_root: Hash,
    /// How many client proofs it covers.
    pub count: u64,
    /// The registry entries it uses.
    pub mask: Vec<bool>,
}

/// A batch built by [`aggregate`].
pub struct AggregatedBatch {
    /// Its encoding (what travels to miners).
    pub proof_bytes: Vec<u8>,
    pub statement: BatchStatement,
    /// Wall time of each step, in order (seconds).
    pub step_seconds: Vec<f64>,
}

/// A batch decided natively ([`decide`]): the miner's pre-pass.
#[derive(Clone, Debug)]
pub struct DecidedBatch {
    pub statement: BatchStatement,
    /// The lanes a block carrying it publishes: its outgoing entry lanes and
    /// its `D_batch` lane after the chain arm's fold.
    pub lanes: ParentLanes,
    /// The chain arm's preparation, what a block producer carries:
    /// computed once, at the decision.
    pub prepared: PreparedBatch,
}

/// A decided batch prepared for the chain arm of a v1.6 block (opaque): the
/// batch proof, the fold of its lincheck into the `D_batch` lane and the
/// recorded transcript. It is a junction on the parent lanes it was decided
/// against, and only a block over those lanes may carry it.
#[derive(Clone)]
pub struct PreparedBatch(pub(crate) Arc<super::chain_arm::PreparedChainBatch>);

impl core::fmt::Debug for PreparedBatch {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("PreparedBatch").finish_non_exhaustive()
    }
}

fn matrices_of(clients: &[AggregationClient]) -> BTreeMap<Hash, Arc<FieldR1cs>> {
    clients
        .iter()
        .map(|client| (client.matrix.structural_statement_digest(), Arc::clone(&client.matrix)))
        .collect()
}

fn step_input<'a>(
    first: ClientBatchFirst<'a>,
    second: Option<&'a ClientBatchProof>,
    lanes: &ParentLanes,
    leaves: &[Hash],
) -> ClientBatchStepInput<'a> {
    ClientBatchStepInput {
        first,
        second,
        incoming_entries: lanes.entries.clone(),
        incoming_self: lanes.batch.clone(),
        leaves: leaves.to_vec(),
    }
}

fn timed_step(
    class: &ClientBatchClass,
    input: &ClientBatchStepInput<'_>,
    matrices: &BTreeMap<Hash, Arc<FieldR1cs>>,
    seconds: &mut Vec<f64>,
) -> Result<ClientBatchProof, String> {
    let started = std::time::Instant::now();
    let proof = step(class, input, matrices)?;
    seconds.push(started.elapsed().as_secs_f64());
    Ok(proof)
}

fn step(
    class: &ClientBatchClass,
    input: &ClientBatchStepInput<'_>,
    matrices: &BTreeMap<Hash, Arc<FieldR1cs>>,
) -> Result<ClientBatchProof, String> {
    let prepared = prepare_client_batch_step(class.identity(), input, matrices)
        .map_err(|error| format!("step preparation: {error:?}"))?;
    let assembled = assemble_client_batch_step(&class.parts, &prepared)
        .map_err(|error| format!("step assembly: {error:?}"))?;
    prove_client_batch_step(class.identity(), &assembled).map_err(|error| format!("step proof: {error:?}"))
}

/// Build one batch of `clients` (in order) under the chain's registry
/// `leaves`, as a junction on `parent`'s lanes. Every client must be of a
/// registered `D` whose matrix is given; each is received (verified,
/// folded) before any step. `held` are the registered matrices the
/// aggregator holds besides its clients': the batch is decided before it is
/// handed out, and its lanes carry the parent's live lanes on, each decided
/// on its own matrix.
pub fn aggregate(
    class: &AggregationClass,
    leaves: &[Hash],
    clients: &[AggregationClient],
    parent: &ParentLanes,
    held: &BTreeMap<Hash, Arc<FieldR1cs>>,
) -> Result<AggregatedBatch, String> {
    if clients.is_empty() {
        return Err("a batch covers at least one client".into());
    }
    let form = &class.form;
    let capacity = form.registry_capacity();
    if leaves.len() != capacity || parent.entries.len() != capacity {
        return Err(format!("the registry and the parent lanes have {capacity} entries"));
    }
    let registry = HistoryStepClientRegistry::new(
        form.registry_depth(),
        leaves.iter().copied().take_while(|leaf| *leaf != [0u8; 32]).collect(),
    )
    .map_err(|error| format!("registry: {error:?}"))?;
    // A class over a hiding form: each distinct matrix authenticated once by
    // the form's matrix set (its `D` computed once, the registration rule
    // applied), every client of it received on that authentication.
    let set = crate::acceptance::history_step::HistoryStepClientMatrixSet::new(form);
    let mut authenticated: Vec<crate::acceptance::history_step::AuthenticatedClientMatrix> = Vec::new();
    let prepared = clients
        .iter()
        .map(|client| {
            if form.zk_layout().is_some() {
                // A class over a hiding form (v1.6): form-Z proofs only.
                let (proof, root, io) = decode_history_step_client_zk_proof(form, &client.proof_bytes)
                    .map_err(|error| format!("client proof: {error:?}"))?;
                if !authenticated.iter().any(|known| Arc::ptr_eq(known.matrix(), &client.matrix)) {
                    authenticated.push(
                        set.authenticate(Arc::clone(&client.matrix))
                            .map_err(|error| format!("client pre-pass: {error:?}"))?,
                    );
                }
                let witness = HistoryStepClientZkWitness {
                    proof,
                    commitment: Commitment {
                        root,
                        params: form.pcs_params().clone(),
                    },
                    io,
                    matrix: Arc::clone(&client.matrix),
                    registry: registry.clone(),
                };
                let matrix = authenticated
                    .iter()
                    .find(|known| Arc::ptr_eq(known.matrix(), &client.matrix))
                    .expect("the client's matrix was authenticated just above");
                return PreparedHistoryStepClient::prepare_z_authenticated(form, &witness, matrix)
                    .map_err(|error| format!("client pre-pass: {error:?}"));
            }
            let (field_proof, root, io) = decode_history_step_client_proof(form, &client.proof_bytes)
                .map_err(|error| format!("client proof: {error:?}"))?;
            let witness = HistoryStepClientWitness {
                field_proof,
                commitment: Commitment {
                    root,
                    params: form.pcs_params().clone(),
                },
                io,
                matrix: Arc::clone(&client.matrix),
                registry: registry.clone(),
            };
            PreparedHistoryStepClient::prepare(form, &witness)
                .map_err(|error| format!("client pre-pass: {error:?}"))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut matrices = held.clone();
    if authenticated.is_empty() {
        matrices.extend(matrices_of(clients));
    } else {
        matrices.extend(authenticated.iter().map(|known| (known.digest(), Arc::clone(known.matrix()))));
    }
    let dead = ParentLanes::empty(capacity);
    let mut step_seconds = Vec::new();
    let junction = if prepared.len() == 1 {
        let input = step_input(ClientBatchFirst::Client(&prepared[0]), None, parent, leaves);
        timed_step(&class.class, &input, &matrices, &mut step_seconds)?
    } else {
        let input = step_input(ClientBatchFirst::Client(&prepared[0]), None, &dead, leaves);
        let mut top = timed_step(&class.class, &input, &matrices, &mut step_seconds)?;
        for client in &prepared[1..] {
            let input = step_input(ClientBatchFirst::Client(client), Some(&top), &dead, leaves);
            let next = timed_step(&class.class, &input, &matrices, &mut step_seconds)?;
            drop(input);
            top = next;
        }
        let input = step_input(ClientBatchFirst::Batch(&top), None, parent, leaves);
        timed_step(&class.class, &input, &matrices, &mut step_seconds)?
    };
    let proof_bytes =
        encode_client_batch_proof(&class.class.parts, &junction).map_err(|error| format!("encode: {error:?}"))?;
    let decided = decide(class, &proof_bytes, leaves, parent, &matrices)?;
    Ok(AggregatedBatch {
        proof_bytes,
        statement: decided.statement,
        step_seconds,
    })
}

/// Decide a batch received as bytes, natively, against the chain's registry
/// `leaves` and the parent's lanes: decoded under the class's bound, the
/// proof and its sidecar verified, its own lincheck on the class matrix, its
/// IO of the class, its leaves the registry, its incoming lanes the
/// parent's, every live lane true on its matrix (each entry's on the
/// registered matrix, the `D_batch` lane on the class matrix); then the
/// chain arm's fold of its lincheck. Whether the entries it uses are
/// carriable at the height of the block that carries it is the chain's rule
/// (`check_batch_mask_carriable`).
pub fn decide(
    class: &AggregationClass,
    proof_bytes: &[u8],
    leaves: &[Hash],
    parent: &ParentLanes,
    matrices: &BTreeMap<Hash, Arc<FieldR1cs>>,
) -> Result<DecidedBatch, String> {
    let proof =
        decode_client_batch_proof(&class.class.parts, proof_bytes).map_err(|error| format!("decode: {error:?}"))?;
    let decision = decide_client_batch(
        &class.class,
        &proof,
        leaves,
        &parent.entries,
        parent.batch.as_ref(),
        matrices,
    )
    .map_err(|error| format!("decision: {error:?}"))?;
    let prepared =
        prepare_chain_batch(&class.class, &proof).map_err(|error| format!("chain arm: {error:?}"))?;
    Ok(DecidedBatch {
        statement: BatchStatement {
            statements_root: decision.root,
            count: decision.count as u64,
            mask: decision.public.mask.clone(),
        },
        lanes: ParentLanes {
            entries: decision.public.out_entries.clone(),
            batch: prepared.folded.clone(),
        },
        prepared: PreparedBatch(Arc::new(prepared)),
    })
}
