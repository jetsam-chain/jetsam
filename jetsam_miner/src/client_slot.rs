// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The v1.5 client slot of a block template (M3.8): which client objects a
//! block carries, with their paying transactions, and which client proof.
//!
//! The node holds what can be carried — client proofs received and
//! pre-passed (`ClientSubmissionQueue`), registrations handed to it — behind
//! [`MinerClientSource`]. A template offers their paying transactions to the
//! ordinary fee selection beside the mempool's; once the selection is final,
//! the objects are exactly the openings of the markers it kept, in transaction
//! order ([`client_slot_of_selection`]), the carried client is the one whose
//! submission was kept, and the registry leaves are the chain's after the
//! kept registration. A payment the selection dropped takes its object (and
//! client) with it: the block can never carry a marker without its object or
//! an object without its marker.

use std::sync::Arc;

use jetsam_chain::block::Block;
use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::client_objects::{
    client_object_marker_kind, client_object_marker_kind_at, queue::ClientCandidate,
    ClientBatchRules, ClientBatchSubmission, ClientObject, ClientObjectKind, ClientObjectRules,
    ClientRegistration, ClientRegistryState, ClientSubmission, CLIENT_REGISTRY_CAPACITY,
};
use jetsam_recursive::acceptance::history_step::PreparedHistoryStepClient;
use jetsam_tx::PagedSpendIntent;

/// What the node holds for its miners.
pub trait MinerClientSource: Send + Sync {
    /// The client the block after `parent` should carry, with its payment
    /// (`anchor_ok`: whether a payment's epoch anchor is minable there).
    fn best_candidate(
        &self,
        parent: &BlockHeader,
        registry: &ClientRegistryState,
        rules: &ClientObjectRules,
        anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    ) -> Option<ClientCandidate<Arc<PreparedHistoryStepClient>>>;

    /// Registrations held for this node's miners, not yet registered, with
    /// the transactions that pay their licenses.
    fn held_registrations(
        &self,
        registry: &ClientRegistryState,
    ) -> Vec<(ClientRegistration, PagedSpendIntent)>;

    /// A block was committed: forget what it settles.
    fn on_block_committed(&self, block: &Block);

    /// The batch the v1.6 block after `parent` should carry: one
    /// decided against exactly the lanes `parent` publishes (read from its
    /// terminal `parent_terminal`; dead at the first v1.6 block) and the
    /// leaves of `registry` (the registry the parent's branch ends with),
    /// with its payment. `None` by default: a source that holds no batches.
    fn best_batch(
        &self,
        _parent: &BlockHeader,
        _parent_terminal: Option<&[u8]>,
        _registry: &ClientRegistryState,
        _anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    ) -> Option<MinerBatch> {
        None
    }

    /// A template refused the block carrying `submission`'s batch: it is not
    /// offered again.
    fn forget_batch(&self, _submission: &ClientBatchSubmission) {}
}

/// A batch a v1.6 block may carry: its submission, the transaction that pays
/// it (and carries its marker), that transaction's fee, and the chain arm's
/// preparation the producer takes.
#[derive(Clone, Debug)]
pub struct MinerBatch {
    pub submission: ClientBatchSubmission,
    pub payment: PagedSpendIntent,
    pub fee: u64,
    pub prepared: jetsam_recursive::PreparedBatch,
}

pub type SharedMinerClientSource = Arc<dyn MinerClientSource>;

/// The client slot of one template.
#[derive(Clone, Default)]
pub struct TemplateClientSlot {
    /// The block's client objects, in the order of their markers.
    pub objects: Vec<ClientObject>,
    /// The client proof the block carries (its submission is in `objects`).
    pub carried: Option<Arc<PreparedHistoryStepClient>>,
    /// The registry leaves the block publishes when it registers a client
    /// (the chain's after the registration, zero-padded); `None` keeps the
    /// parent's.
    pub registry_leaves: Option<Vec<[u8; 32]>>,
    /// The batch submission the block carries (v1.6): its payment kept by
    /// the selection.
    pub batch_submission: Option<ClientBatchSubmission>,
    /// The batch the block carries (v1.6), set by the template when the
    /// kept batch submission is the one it offered.
    pub batch: Option<jetsam_recursive::PreparedBatch>,
}

impl std::fmt::Debug for TemplateClientSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemplateClientSlot")
            .field("objects", &self.objects)
            .field("carried", &self.carried.as_ref().map(|client| client.digest()))
            .field("registry_leaves", &self.registry_leaves.is_some())
            .field("batch_submission", &self.batch_submission)
            .field("batch", &self.batch.is_some())
            .finish()
    }
}

/// The objects the payments a template offered may open, and the client a
/// kept submission carries.
#[derive(Clone, Default)]
pub struct OfferedClientObjects {
    pub objects: Vec<ClientObject>,
    pub candidate: Option<(ClientSubmission, Arc<PreparedHistoryStepClient>)>,
}

/// The client slot of a final selection: `selected` is the block's user
/// pages in order, `offered` what the template offered, `registry` the
/// parent's registry.
pub fn client_slot_of_selection(
    selected: &[jetsam_tx::Transaction],
    offered: &OfferedClientObjects,
    registry: &ClientRegistryState,
) -> TemplateClientSlot {
    slot_of_selection(selected, offered, registry, |owner| client_object_marker_kind(owner))
}

/// [`client_slot_of_selection`] for a block at `height`: from the v1.6
/// height on, a batch submission marker opens the offered batch submission
/// too ([`client_object_marker_kind_at`]), which becomes the slot's
/// `batch_submission`.
pub fn client_slot_of_selection_at(
    selected: &[jetsam_tx::Transaction],
    offered: &OfferedClientObjects,
    registry: &ClientRegistryState,
    height: u64,
    batch: &ClientBatchRules,
) -> TemplateClientSlot {
    slot_of_selection(selected, offered, registry, |owner| {
        client_object_marker_kind_at(owner, height, batch)
    })
}

/// `slot` for a block at `height`, publishing at the v1.6 relation's base
/// the registry it inherits. That base (the block at the v1.6 height) has no
/// parent proof, so a slot that registers nothing — whose leaves default to
/// the parent's — would publish an empty registry, which the chain refuses
/// whenever the registry holds an entry (`check_terminal_client_view`
/// compares the leaves with the registry). The circuit leaves a base's leaves
/// to the node; above the base it keeps the parent's.
pub fn with_registry_at_v1_6_base(
    mut slot: TemplateClientSlot,
    registry: &ClientRegistryState,
    height: u64,
    batch: &ClientBatchRules,
) -> TemplateClientSlot {
    if slot.registry_leaves.is_none() && batch.activation_height == Some(height) && !registry.is_empty() {
        let mut leaves = registry.digests();
        leaves.resize(CLIENT_REGISTRY_CAPACITY.max(leaves.len()), [0u8; 32]);
        slot.registry_leaves = Some(leaves);
    }
    slot
}

fn slot_of_selection(
    selected: &[jetsam_tx::Transaction],
    offered: &OfferedClientObjects,
    registry: &ClientRegistryState,
    marker_kind: impl Fn(&jetsam_poseidon2b::primitives::Address) -> Option<ClientObjectKind>,
) -> TemplateClientSlot {
    let mut slot = TemplateClientSlot::default();
    for (_, output) in selected.iter().flat_map(|tx| tx.body.live_outputs()) {
        if marker_kind(&output.owner).is_none() {
            continue;
        }
        let Some(object) = offered
            .objects
            .iter()
            .find(|object| object.marker() == output.owner)
        else {
            continue;
        };
        if slot.objects.contains(object) {
            continue;
        }
        slot.objects.push(*object);
    }
    let mut leaves = registry.digests();
    let mut registers = false;
    for object in &slot.objects {
        match object {
            ClientObject::Registration(registration) => {
                leaves.push(registration.matrix_digest);
                registers = true;
            }
            ClientObject::Submission(submission) => {
                slot.carried = offered
                    .candidate
                    .as_ref()
                    .filter(|(candidate, _)| candidate == submission)
                    .map(|(_, client)| Arc::clone(client));
            }
            // v1.6: the template sets the batch itself when the kept
            // submission is the one it offered.
            ClientObject::BatchSubmission(batch) => slot.batch_submission = Some(*batch),
        }
    }
    if registers {
        leaves.resize(CLIENT_REGISTRY_CAPACITY.max(leaves.len()), [0u8; 32]);
        slot.registry_leaves = Some(leaves);
    }
    slot
}

#[cfg(test)]
mod tests;
