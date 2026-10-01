// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The miner's choice (M3 task 3.6): which received client proof the next
//! block carries.
//!
//! A node receives client proofs with their submission and paying
//! transaction (M3.7 transport), pre-passes each one once on reception
//! (`PreparedHistoryStepClient`, outside the block's critical path) and keeps
//! the result here, as the opaque payload `T`. The template of the next block
//! asks [`ClientSubmissionQueue::best`]: at most one client per block (the
//! relation has one client arm), the highest paying fee among those the block
//! may carry, the earliest received on a tie; none means the block pays the
//! ghost, as every block without a client does.

use std::collections::{BTreeMap, BTreeSet};

use jetsam_poseidon2b::primitives::Digest;
use jetsam_tx::PagedSpendIntent;

use super::{
    client_object_marker_kind, ClientObject, ClientObjectRules, ClientRegistryState,
    ClientSubmission,
};
use crate::block::Block;
use crate::block_header::BlockHeader;
use crate::consensus::fees::fee_breakdown;

/// One received client proof, ready to be carried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientCandidate<T> {
    pub submission: ClientSubmission,
    /// The ordinary transaction that pays the submission fee: its single
    /// marker opens `submission`.
    pub payment: PagedSpendIntent,
    /// What the miner needs to carry the client (the pre-passed proof).
    pub payload: T,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientQueueError {
    /// The payment is not a canonical logical transaction.
    MalformedPayment,
    /// The payment's marker does not open this submission (none, several,
    /// another object, or a marker carrying value).
    NotThePayment,
    /// The payment's fee is below the submission fee alone.
    FeeTooLow { minimum: u64, paid: u64 },
    /// This client proof (`D`, `io_commitment`) is already queued.
    Duplicate,
    /// The queue is full of candidates paying at least as much.
    Full,
}

#[derive(Clone, Debug)]
struct Queued<T> {
    candidate: ClientCandidate<T>,
    fee: u64,
    live_inputs: u16,
    live_outputs: u16,
    anchor: Digest,
    arrival: u64,
}

impl<T> Queued<T> {
    /// The worse of two candidates: lower fee, then later arrival.
    fn rank(&self) -> (u64, std::cmp::Reverse<u64>) {
        (self.fee, std::cmp::Reverse(self.arrival))
    }

    fn spends_any(&self, slots: &BTreeSet<u32>) -> bool {
        self.candidate
            .payment
            .pages
            .iter()
            .flat_map(|page| page.body.live_inputs())
            .any(|(_, input)| slots.contains(&input.slot_index))
    }
}

/// Received candidates, bounded, keyed by client proof (`D`, `io_commitment`).
#[derive(Clone, Debug)]
pub struct ClientSubmissionQueue<T> {
    capacity: usize,
    next_arrival: u64,
    entries: BTreeMap<(Digest, Digest), Queued<T>>,
}

impl<T> ClientSubmissionQueue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            next_arrival: 0,
            entries: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Queue one candidate. When full, the lowest-paying queued candidate
    /// (the latest received among equals) makes room for a strictly better
    /// one and is returned.
    pub fn insert(
        &mut self,
        candidate: ClientCandidate<T>,
        rules: &ClientObjectRules,
    ) -> Result<Option<ClientCandidate<T>>, ClientQueueError> {
        let spend = jetsam_tx::validate_paged_spend(&candidate.payment.pages)
            .map_err(|_| ClientQueueError::MalformedPayment)?;
        let marker = ClientObject::Submission(candidate.submission).marker();
        let markers: Vec<_> = candidate
            .payment
            .pages
            .iter()
            .flat_map(|page| page.body.live_outputs())
            .filter(|(_, output)| client_object_marker_kind(&output.owner).is_some())
            .map(|(_, output)| (output.owner, output.amount))
            .collect();
        if markers != [(marker, 0)] {
            return Err(ClientQueueError::NotThePayment);
        }
        if spend.fee < rules.submission_fee_micro {
            return Err(ClientQueueError::FeeTooLow {
                minimum: rules.submission_fee_micro,
                paid: spend.fee,
            });
        }
        let key = (
            candidate.submission.matrix_digest,
            candidate.submission.io_commitment,
        );
        if self.entries.contains_key(&key) {
            return Err(ClientQueueError::Duplicate);
        }
        let arrival = self.next_arrival;
        let queued = Queued {
            candidate,
            fee: spend.fee,
            live_inputs: spend.live_inputs,
            live_outputs: spend.live_outputs,
            anchor: spend.epoch_anchor,
            arrival,
        };
        let mut evicted = None;
        if self.entries.len() >= self.capacity {
            let worst = self
                .entries
                .iter()
                .min_by_key(|(_, queued)| queued.rank())
                .map(|(key, queued)| (*key, queued.rank()))
                .expect("a full queue is not empty");
            if queued.rank() <= worst.1 {
                return Err(ClientQueueError::Full);
            }
            evicted = self.entries.remove(&worst.0).map(|queued| queued.candidate);
        }
        self.next_arrival += 1;
        self.entries.insert(key, queued);
        Ok(evicted)
    }

    /// The candidate the block at `height` (child of `parent`) should carry:
    /// its client carriable at `height`, its payment minable (`anchor_ok` on
    /// its epoch anchor) and paying at least the required fee plus the
    /// submission fee; the highest fee, then the earliest received.
    pub fn best(
        &self,
        parent: &BlockHeader,
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        anchor_ok: impl Fn(&Digest) -> bool,
    ) -> Option<&ClientCandidate<T>> {
        let height = parent.height.saturating_add(1);
        self.entries
            .values()
            .filter(|queued| {
                registry
                    .check_carriable(&queued.candidate.submission.matrix_digest, height)
                    .is_ok()
            })
            .filter(|queued| anchor_ok(&queued.anchor))
            .filter(|queued| {
                let required = fee_breakdown(
                    u64::from(queued.live_inputs),
                    u64::from(queued.live_outputs),
                    parent.active_slot_count,
                    parent.log_slots,
                )
                .required_total
                .saturating_add(rules.submission_fee_micro);
                queued.fee >= required
            })
            .max_by_key(|queued| queued.rank())
            .map(|queued| &queued.candidate)
    }

    /// Forget what a committed block settles: the submission it paid, and
    /// every candidate whose payment spends an input the block spent.
    pub fn on_block_committed(&mut self, block: &Block) -> usize {
        let before = self.entries.len();
        for object in &block.client_objects {
            if let ClientObject::Submission(submission) = object {
                self.remove(submission);
            }
        }
        let spent: BTreeSet<u32> = block
            .transactions
            .iter()
            .flat_map(|tx| tx.body.live_inputs())
            .map(|(_, input)| input.slot_index)
            .collect();
        self.entries.retain(|_, queued| !queued.spends_any(&spent));
        before - self.entries.len()
    }

    /// Forget candidates whose payment can no longer be mined.
    pub fn retain_minable(&mut self, anchor_ok: impl Fn(&Digest) -> bool) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, queued| anchor_ok(&queued.anchor));
        before - self.entries.len()
    }

    pub fn remove(&mut self, submission: &ClientSubmission) -> Option<ClientCandidate<T>> {
        self.entries
            .remove(&(submission.matrix_digest, submission.io_commitment))
            .map(|queued| queued.candidate)
    }
}

#[cfg(test)]
mod tests;
