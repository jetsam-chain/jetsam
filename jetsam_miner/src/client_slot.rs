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
    client_object_marker_kind, queue::ClientCandidate, ClientObject, ClientObjectRules,
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
}

impl std::fmt::Debug for TemplateClientSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemplateClientSlot")
            .field("objects", &self.objects)
            .field("carried", &self.carried.as_ref().map(|client| client.digest()))
            .field("registry_leaves", &self.registry_leaves.is_some())
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
    let mut slot = TemplateClientSlot::default();
    for (_, output) in selected.iter().flat_map(|tx| tx.body.live_outputs()) {
        if client_object_marker_kind(&output.owner).is_none() {
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
