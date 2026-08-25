// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 trace.protocol.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Research imports the production PagedSpend primitive and stream validator.
//! No consensus candidate is duplicated here.

pub use elide_chain::consensus::paged_spend::{
    validate_paged_spend_tx_page_stream as validate_paged_spend_stream,
    validate_paged_spend_tx_page_stream_for_class as validate_paged_spend_stream_for_class,
    BlockProofClass as ProofClass, PagedSpendGroupFacts, PagedSpendStreamError,
    PagedSpendStreamFacts,
};
pub use elide_tx::{
    canonical_paged_spend_auth, hash_paged_spend, validate_paged_spend, CanonicalPagedSpendAuth,
    PagedSpendError, PagedSpendFacts, PagedSpendIntent, TxPage, MAX_PAGED_SPEND_INPUTS,
    MAX_PAGED_SPEND_INTENT_BYTES, MAX_PAGED_SPEND_OUTPUTS, MAX_PAGED_SPEND_PAGES,
    PAGED_SPEND_END_BIT, PAGED_SPEND_START_BIT,
};

pub const MAX_BLOCK_USER_PAGES: usize = ProofClass::B255.page_capacity();
pub const MAX_BLOCK_LOGICAL_TRANSACTIONS: usize = ProofClass::B255.live_authorization_capacity();
pub const MAX_BLOCK_LIVE_INPUTS: usize = ProofClass::B255.input_capacity();
pub const MAX_BLOCK_LIVE_OUTPUTS: usize = ProofClass::B255.output_capacity();
