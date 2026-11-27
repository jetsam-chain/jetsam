// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T3.7): the guardian slot of a block template.
//!
//! The mempool holds guardian intents with their transactions
//! (`jetsam_mempool::guardian`), judged at admission and at each tip. A
//! template places them after the destination objects, in the order of T1.1
//! §10 ([`plan_guardian_offer`]): the equivocation, the highest attestation,
//! then the delegation objects by arrival, within the section's 32 objects
//! and byte budget; before the latch the reward fundings only by pairs, one
//! of each development fund, or none (F-4). Their transactions go to the fee
//! selection with the others. Once built, the block's guardian objects are
//! the offer's whose transactions the block kept, in the order of their
//! markers ([`GuardianOffer::objects_for`]); the template judges the block by
//! `validate_block_guardian` and, refused, builds it again without the
//! guardian objects, forgetting those the refusal blames ([`culprits_of`]).
//! No guardian object is urgent: no new trigger wakes the miner.

use jetsam_chain::consensus::destination::{
    destination_section_wire_len, DestinationObject, DestinationSection,
    MAX_BLOCK_DESTINATION_OBJECTS,
};
use jetsam_chain::consensus::guardian::{
    guardian_marker_kind, GuardianBranch, GuardianError, GuardianObjectKind, GuardianRules,
    GuardianState, RewardFund,
};
use jetsam_chain::consensus::wire_limits::MAX_DESTINATION_SECTION_BYTES;
use jetsam_mempool::guardian::{apply_guardian_intent, project, HeldGuardian, Signatures};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{PagedSpendIntent, Transaction};

/// Bytes an object takes in the section besides its body: tag and length.
const OBJECT_HEADER_BYTES: usize = 1 + 4;

/// What a template offers to carry of the guardian intents, in placement
/// order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuardianOffer {
    pub held: Vec<HeldGuardian>,
}

impl GuardianOffer {
    pub fn is_empty(&self) -> bool {
        self.held.is_empty()
    }

    /// The transactions the offer brings, in placement order.
    pub fn payments(&self) -> impl Iterator<Item = &PagedSpendIntent> {
        self.held.iter().map(|held| &held.intent.payment)
    }

    /// Their pages as transactions, in [`Self::payments`] order.
    pub fn payment_order(&self) -> Vec<Transaction> {
        self.payments()
            .flat_map(|payment| payment.pages.iter())
            .map(|page| Transaction::new(page.body.clone()))
            .collect()
    }

    /// The guardian objects of a block whose transactions are `txs`: the
    /// offer's objects whose markers the block carries, in the order of the
    /// markers (M-6).
    pub fn objects_for(&self, txs: &[Transaction]) -> Vec<DestinationObject> {
        txs.iter()
            .flat_map(|tx| {
                tx.body
                    .live_outputs()
                    .map(|(_, output)| output.owner)
                    .collect::<Vec<Address>>()
            })
            .filter(|owner| guardian_marker_kind(owner).is_some())
            .filter_map(|marker| {
                self.held
                    .iter()
                    .find(|held| held.intent.object.marker() == marker)
                    .map(|held| held.intent.object.object())
            })
            .collect()
    }

    /// The arrival numbers of what it offers.
    pub fn seqs(&self) -> Vec<u64> {
        self.held.iter().map(|held| held.seq).collect()
    }
}

fn funder(held: &HeldGuardian) -> Option<Address> {
    (held.intent.object.kind == GuardianObjectKind::RewardFund)
        .then(|| RewardFund::decode(&held.intent.object.body).ok())
        .flatten()
        .map(|fund| fund.funder)
}

/// What the block at `height` after a branch in `state` (the branch
/// `branch`) offers, from the intents the pool holds (`candidates`, in its
/// order: the equivocation, the attestation, the delegation objects by
/// arrival), after the `destination` objects already in its section, each
/// only if the pool's judge takes it after those placed before it (A3-7).
/// `anchor_ok` says whether a transaction's epoch anchor is minable in that
/// block.
pub fn plan_guardian_offer(
    state: &GuardianState,
    rules: &GuardianRules,
    height: u64,
    branch: &dyn GuardianBranch,
    candidates: Vec<HeldGuardian>,
    anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    destination: &DestinationSection,
) -> GuardianOffer {
    let budget = MAX_BLOCK_DESTINATION_OBJECTS.saturating_sub(destination.objects.len());
    let mut bytes = destination_section_wire_len(destination);
    let usable: Vec<HeldGuardian> = candidates
        .into_iter()
        .filter(|held| {
            held.intent
                .payment
                .pages
                .first()
                .is_some_and(|page| anchor_ok(&page.body.epoch_anchor))
        })
        .collect();
    let latched = state.activated_at.is_some();
    // Before the latch, the first funding of each fund comes first: a pair
    // or nothing (F-4).
    let ordered = if latched {
        usable
    } else {
        let first_of = |fund: Address| usable.iter().position(|held| funder(held) == Some(fund));
        match (first_of(rules.network_fund), first_of(rules.lab_fund)) {
            (Some(network), Some(lab)) => {
                let (fundings, others): (Vec<_>, Vec<_>) = usable
                    .iter()
                    .enumerate()
                    .partition(|(_, held)| funder(held).is_some());
                let mut ordered: Vec<HeldGuardian> =
                    others.into_iter().map(|(_, held)| held.clone()).collect();
                ordered.push(usable[network].clone());
                ordered.push(usable[lab].clone());
                ordered.extend(
                    fundings
                        .into_iter()
                        .filter(|(index, _)| *index != network && *index != lab)
                        .map(|(_, held)| held.clone()),
                );
                ordered
            }
            _ => usable
                .into_iter()
                .filter(|held| funder(held).is_none())
                .collect(),
        }
    };
    let mut offer = GuardianOffer::default();
    // A3-7: each object as the pool's judge takes it after those placed
    // before it (signatures aside: verified at admission), so that none is
    // offered beside one that voids it (an attestation after the
    // equivocation that revokes the key, A-3).
    let (mut projected, _) = project(state, height, std::iter::empty(), branch, rules);
    for held in ordered {
        if offer.held.len() >= budget {
            break;
        }
        let len = OBJECT_HEADER_BYTES + held.intent.object.body.len();
        if bytes + len > MAX_DESTINATION_SECTION_BYTES {
            continue;
        }
        if apply_guardian_intent(
            &mut projected,
            height,
            &held.group,
            &held.intent.object,
            branch,
            rules,
            Signatures::Defer,
        )
        .is_err()
        {
            continue;
        }
        bytes += len;
        offer.held.push(held);
    }
    // Before the latch nothing else a block may carry depends on a funding
    // (the delegation objects wait for the latch): a half pair leaves alone.
    if !latched {
        let has = |fund: Address| offer.held.iter().any(|held| funder(held) == Some(fund));
        if !(has(rules.network_fund) && has(rules.lab_fund)) {
            offer.held.retain(|held| funder(held).is_none());
        }
    }
    offer
}

/// What a block carrying `offer`, refused with `error` by
/// `validate_block_guardian` against the parent's `state`, blames among the
/// guardian objects: the ones the pool's judge refuses in placement order
/// (the block's rules, signatures aside); if none, for a bad signature the
/// attestation and the equivocation offered. Nothing for what the template
/// itself decided (counts, F-4) or a local fault: the block is then built
/// without the guardian objects and they stay held.
pub fn culprits_of(
    error: &GuardianError,
    offer: &GuardianOffer,
    state: &GuardianState,
    height: u64,
    branch: &dyn GuardianBranch,
    rules: &GuardianRules,
) -> Vec<u64> {
    let (_, refused) = project(state, height, &offer.held, branch, rules);
    if !refused.is_empty() {
        return refused.into_iter().map(|(seq, _)| seq).collect();
    }
    if *error == GuardianError::InvalidGuardianSignature {
        return offer
            .held
            .iter()
            .filter(|held| {
                matches!(
                    held.intent.object.kind,
                    GuardianObjectKind::Attestation | GuardianObjectKind::Equivocation
                )
            })
            .map(|held| held.seq)
            .collect();
    }
    Vec::new()
}

#[cfg(test)]
mod tests;
