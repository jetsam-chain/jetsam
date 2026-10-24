// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Recursive proof authority for one atomic accepted block.
//!
//! The only protocol carried by this crate is `HistoryStep`: the current
//! block relation and the persisted parent terminal are proved as one unit.

pub mod acceptance;
pub mod accumulator;
pub mod region_sidecar;

pub use acceptance::development_payout_pins::{
    development_payout_owner_pins, development_payout_owner_pins_for,
    verify_relation_development_payout_pins, DevelopmentPayoutPinError, DevelopmentPayoutPinScan,
    DevelopmentPayoutRelationError, DEVELOPMENT_PAYOUT_PIN_COUNT, DEVELOPMENT_PAYOUT_PIN_LABELS,
};
pub use acceptance::history_step::{
    decode_history_step_terminal, decode_verify_history_step_terminal,
    decode_verify_history_step_terminal_rooted, derive_history_step_direct_block_vk,
    derive_history_step_direct_block_vk_in, derive_history_step_runtime_parts,
    derive_history_step_runtime_parts_in, encode_history_step_terminal,
    freeze_history_step_bank, freeze_history_step_bank_in,
    history_step_terminal_max_wire_bytes, history_step_terminal_wire_bytes,
    pin_history_step_class_bank,
    prepare_history_step_authorizations, prepare_history_step_authorizations_in,
    prepare_history_step_for_pow, prepare_history_step_ghost_authorization,
    prove_built_history_step_terminal, prove_history_step, verify_history_step_terminal,
    verify_history_step_terminal_rooted, AcceptedHistoryStepTerminal,
    AuthorizationComponentInput, BuiltHistoryStep, ExactStateStructuralFrontierInputs,
    FrozenHistoryStepBank, HistoryStepAuthorizationError, HistoryStepBlockComponents,
    HistoryStepBlockInput, HistoryStepError, HistoryStepFreezeError, HistoryStepFreezeInput,
    HistoryStepFreezeInputProvider, HistoryStepFreezeMatrixStore, HistoryStepFreezeStage,
    HistoryStepInputError, HistoryStepMatrixLease, HistoryStepMatrixSource,
    HistoryStepMatrixSourceError, HistoryStepParentTranscriptLayout, HistoryStepRuntime,
    HistoryStepRuntimeParts, HistoryStepSidecarOperation, HistoryStepTerminal,
    PreparedHistoryStepAuthorizations, PreparedHistoryStepForPow,
    PreparedHistoryStepGhostAuthorization, HISTORY_STEP_RUNTIME_PARTS_COMPACT_CLIENT_VERSION,
    HISTORY_STEP_RUNTIME_PARTS_COMPACT_MAX_BYTES, HISTORY_STEP_RUNTIME_PARTS_COMPACT_VERSION,
    HISTORY_STEP_WIRE_VERSION,
};
pub use acceptance::history_step_bank::{
    canonical_history_step_class_id, canonical_history_step_class_id_in,
    canonical_history_step_pcs_params, canonical_history_step_pcs_params_in,
    canonical_history_step_shape, canonical_history_step_shape_in, history_step_class_ms,
    history_step_bank_io_layout, history_step_bank_io_layout_for,
    history_step_bank_io_spec, history_step_bank_io_spec_for, CanonicalHistoryStepClassId,
    HistoryStepBankEntryPins, HistoryStepBankError, HistoryStepBankIoLayout,
    PinnedHistoryStepBankEntry, PinnedHistoryStepClassBank, RecursionRoot,
    HISTORY_STEP_CLASS_COUNT, HISTORY_STEP_CURRENT_CLASS_MS, HISTORY_STEP_TIER_SLOT_COUNT,
    V1_3_RECURSION_ROOT_LANES,
};
/// The v1.5 client slot.
#[doc(hidden)]
pub use acceptance::history_step::{measure_history_step_v1_6_rows, HistoryStepV16RowsGate};
pub use acceptance::history_step::{
    check_history_step_client_lanes, decode_verify_history_step_terminal_rooted_with_client_matrices,
    verify_history_step_terminal_rooted_with_client_matrices, AuthenticatedClientMatrix,
    HistoryStepClientMatrices, HistoryStepClientMatrixSet,
};
pub use acceptance::history_step::{
    client_io_commitment, decode_history_step_client_proof, decode_history_step_client_zk_proof,
    encode_history_step_client_proof, encode_history_step_client_zk_proof,
    history_step_client_proof_max_wire_bytes, history_step_client_zk_proof_max_wire_bytes,
    HISTORY_STEP_CLIENT_ZK_PROOF_WIRE_VERSION, prepare_history_step_for_pow_with_batch, prepare_history_step_for_pow_with_client, HistoryStepChainClients, HistoryStepClientParts,
    HistoryStepClientCarry, HistoryStepClientRegistry, HistoryStepClientWitness,
    HistoryStepClientZkWitness, PreparedHistoryStepClient,
    HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};
pub use acceptance::history_step_bank::{
    parse_history_step_client_lanes, HistoryStepCarriedClient, HistoryStepClientClaim,
    HistoryStepClientForm, HistoryStepClientIoLanes, HISTORY_STEP_CLIENT_FORM_M,
};
/// The aggregator's side of the v1.6 batch slot: the
/// `jetsam-aggregate` tool and the miner's pre-pass. The batch class is an
/// input until it is frozen as a consensus constant.
#[doc(hidden)]
pub use acceptance::client_batch::aggregate::{
    aggregate as aggregate_client_batch, decide as decide_client_batch_bytes, AggregatedBatch,
    AggregationClass, AggregationClient, BatchStatement, DecidedBatch, ParentLanes, PreparedBatch,
    terminal_parent_lanes as history_step_terminal_parent_lanes,
};
/// The v1.6 batch slot: the batch lanes a node reads and decides.
pub use acceptance::history_step_bank::{
    parse_history_step_batch_lanes, HistoryStepBatchClaim, HistoryStepBatchClassPins,
    HistoryStepBatchIoLanes, HistoryStepBatchMatrix, HistoryStepCarriedBatch,
    HISTORY_STEP_BATCH_MAX_CLIENTS,
};
pub use accumulator::{
    genesis_accumulator, ChainAccumulator, ChainAccumulatorAdvanceError, ChainAccumulatorLaneError,
    ChainAccumulatorLocalBoundaryError, CHAIN_ACCUMULATOR_LANES,
};
/// The pack generation vocabulary, re-exported so that a crate proving or
/// verifying a HistoryStep never has to reach into the consensus crate for
/// the one enum the generation-aware signatures here are parameterised by.
pub use jetsam_chain::consensus::params::HistoryStepPackGeneration;
