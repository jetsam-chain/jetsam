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
    PreparedHistoryStepGhostAuthorization, HISTORY_STEP_RUNTIME_PARTS_COMPACT_MAX_BYTES,
    HISTORY_STEP_RUNTIME_PARTS_COMPACT_VERSION, HISTORY_STEP_WIRE_VERSION,
};
pub use acceptance::history_step_bank::{
    canonical_history_step_class_id, canonical_history_step_class_id_in,
    canonical_history_step_pcs_params, canonical_history_step_shape,
    history_step_bank_io_layout, history_step_bank_io_layout_for,
    history_step_bank_io_spec, history_step_bank_io_spec_for, CanonicalHistoryStepClassId,
    HistoryStepBankEntryPins, HistoryStepBankError, HistoryStepBankIoLayout,
    PinnedHistoryStepBankEntry, PinnedHistoryStepClassBank, RecursionRoot,
    HISTORY_STEP_CLASS_COUNT, HISTORY_STEP_CURRENT_CLASS_MS, HISTORY_STEP_TIER_SLOT_COUNT,
    V1_3_RECURSION_ROOT_LANES,
};
pub use accumulator::{
    genesis_accumulator, ChainAccumulator, ChainAccumulatorAdvanceError, ChainAccumulatorLaneError,
    ChainAccumulatorLocalBoundaryError, CHAIN_ACCUMULATOR_LANES,
};
/// The pack generation vocabulary, re-exported so that a crate proving or
/// verifying a HistoryStep never has to reach into the consensus crate for
/// the one enum the generation-aware signatures here are parameterised by.
pub use jetsam_chain::consensus::params::HistoryStepPackGeneration;
