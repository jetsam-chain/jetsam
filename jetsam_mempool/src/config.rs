// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Mempool configuration.

use jetsam_chain::consensus::client_objects::ClientObjectRules;
use jetsam_chain::consensus::wire_limits::{MAX_MEMPOOL_BYTES, MAX_MEMPOOL_TXS};

/// Configuration for the async mempool.
#[derive(Debug, Clone)]
pub struct MempoolConfig {
    /// Maximum number of admitted transactions.
    pub capacity: usize,

    /// Maximum serialized PagedSpendIntent bytes retained in RAM.
    pub max_total_intent_bytes: usize,

    /// Number of recent admitted-tx fees used to compute the dynamic fee floor.
    /// Floor = max(MIN_FEE_BASE, median(last N fees) × 0.9).
    pub fee_floor_window: usize,

    /// Number of concurrent authorization verification workers (`spawn_blocking` slots).
    /// 0 = no concurrency limit; authorization verification is still required.
    /// Recommended: number of physical cores.
    pub auth_verify_workers: usize,

    /// The v1.5 client-object rules the pool judges plain transactions by
    /// (`check_plain_transaction`): the consensus rules of this binary. A
    /// test arms a clock of its own with [`Self::with_client_object_rules`].
    pub client_object_rules: ClientObjectRules,
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            capacity: MAX_MEMPOOL_TXS,
            max_total_intent_bytes: MAX_MEMPOOL_BYTES,
            fee_floor_window: 50,
            auth_verify_workers: 4,
            client_object_rules: ClientObjectRules::current(),
        }
    }
}

impl MempoolConfig {
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }

    pub fn with_max_total_intent_bytes(mut self, bytes: usize) -> Self {
        self.max_total_intent_bytes = bytes;
        self
    }

    pub fn with_auth_verify_workers(mut self, n: usize) -> Self {
        self.auth_verify_workers = n;
        self
    }

    pub fn with_client_object_rules(mut self, rules: ClientObjectRules) -> Self {
        self.client_object_rules = rules;
        self
    }
}
