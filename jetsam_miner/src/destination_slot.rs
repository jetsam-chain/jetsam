// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T2.5): the destination slot of a block template.
//!
//! The mempool holds destination intents with their transactions
//! (`jetsam_mempool::destination`). A template offers what the branch can
//! use ([`plan_destination_offer`]): the committee update it needs next, and
//! releases by increasing `n` among the payable ones (porte R §3.3: never
//! jumping one that waits only for the V6 window), sharing one finality
//! update, within the section's object cap (32 objects in all, T2.2) and byte
//! budget. Their transactions go to the fee selection with the mempool's;
//! mempool deposits that would take the vault over its ceiling in the order
//! the chain template places the groups are left out
//! ([`deposits_within_ceiling`], A1 bis n-6). Once built, the block's section
//! is the offer's objects whose transactions the block kept, releases in the
//! order of their markers ([`DestinationOffer::section_for`]); the template
//! judges the block by `validate_block_destination` and, refused, builds it
//! again without what the refusal blames ([`retry_without_culprits`], A2-M1),
//! then, still refused, without the destination objects, forgetting those the
//! refusals blame ([`culprits_of`]).
//!
//! From the return-over-Base height on (D101, D107; dormant on every profile
//! today): releases are offered over Base only (their game proof and output
//! root preimage shared like the finality update), each paying what V2 asks
//! of the branch (the creditor of the advance of exactly its burn, D114);
//! never in a block that carries a committee update (D107 §9.1: the update
//! goes first, the releases wait one block); and the mempool's advances and
//! claim transfers are placed only as the block rule accepts them together
//! ([`advances_within_rules`]).

use jetsam_chain::consensus::destination::advance::{release_payee, AdvanceCheck};
use jetsam_chain::consensus::destination::state::{
    committee_update_marker, DestinationBlockError, DestinationRules, DestinationState,
};
use jetsam_chain::consensus::destination::{
    destination_section_wire_len, DestinationObject, DestinationObjectKind, DestinationSection,
    MAX_BLOCK_DESTINATION_OBJECTS,
};
use jetsam_chain::consensus::vault::{
    destination_marker_kind, DestinationMarkerKind, VaultError, VaultRules,
};
use jetsam_chain::consensus::wire_limits::MAX_DESTINATION_SECTION_BYTES;
use jetsam_chain::consensus::destination::fast::{fast_marker_kind, FastMarkerKind};
use jetsam_mempool::destination::{
    check_object_payment_vault, CommitteeUpdateIntent, FastContradictionIntent,
    FastRevocationIntent, ReleaseIntent,
};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{PagedSpendIntent, Transaction, TxPage};

/// What a template offers to carry in its destination section.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DestinationOffer {
    pub committee_update: Option<CommitteeUpdateIntent>,
    /// By increasing `n`, all proven under the first one's finality update
    /// (and, over Base, its game proof and preimage; fast releases its
    /// attestation: one mode per block).
    pub releases: Vec<ReleaseIntent>,
    /// The fast lane's contradictions (never beside a fast release), under
    /// the Base objects of the releases, or of the first contradiction.
    pub contradictions: Vec<FastContradictionIntent>,
    /// The cold key's revocation of the relay's key.
    pub revocation: Option<FastRevocationIntent>,
}

impl DestinationOffer {
    pub fn is_empty(&self) -> bool {
        self.committee_update.is_none()
            && self.releases.is_empty()
            && self.contradictions.is_empty()
            && self.revocation.is_none()
    }

    /// The section objects the offer makes when all of it is carried: the
    /// shared proofs once (finality update and account proof, and over Base
    /// the game proof and output root preimage, and a fast release's
    /// attestation), then one per release and contradiction; the
    /// revocation.
    pub fn object_count(&self) -> usize {
        let shared = match (self.releases.first(), self.contradictions.first()) {
            (Some(first), _) => {
                (if first.base.is_some() { 4 } else { 2 }) + usize::from(first.fast.is_some())
            }
            (None, Some(_)) => 4,
            (None, None) => 0,
        };
        usize::from(self.committee_update.is_some())
            + shared
            + self.releases.len()
            + self.contradictions.len()
            + usize::from(self.revocation.is_some())
    }

    /// The transactions the offer brings, committee update first.
    pub fn payments(&self) -> impl Iterator<Item = &PagedSpendIntent> {
        self.committee_update
            .iter()
            .map(|update| &update.payment)
            .chain(self.releases.iter().map(|release| &release.payment))
            .chain(self.contradictions.iter().map(|c| &c.payment))
            .chain(self.revocation.iter().map(|r| &r.payment))
    }

    /// Their pages as transactions, in [`Self::payments`] order.
    pub fn payment_order(&self) -> Vec<Transaction> {
        self.payments()
            .flat_map(|payment| payment.pages.iter())
            .map(|page| Transaction::new(page.body.clone()))
            .collect()
    }

    /// The section of a block whose transactions are `txs`: the committee
    /// update if its marker is there; the finality update and account proof
    /// once (over Base the game proof and preimage, and a fast release's
    /// attestation), then the release objects in the order of their markers,
    /// if any release group is there (else the contradictions' proofs, if any
    /// contradiction is there); the contradictions in the order of their
    /// markers; the revocation if its marker is there.
    pub fn section_for(&self, txs: &[Transaction]) -> DestinationSection {
        let owners: Vec<Address> = txs
            .iter()
            .flat_map(|tx| {
                tx.body
                    .live_outputs()
                    .map(|(_, output)| output.owner)
                    .collect::<Vec<_>>()
            })
            .collect();
        let markers: Vec<Address> = owners
            .iter()
            .copied()
            .filter(|owner| destination_marker_kind(owner).is_some())
            .collect();
        let mut objects = Vec::new();
        if let Some(update) = &self.committee_update {
            if markers.contains(&committee_update_marker(&update.payload)) {
                objects.push(update.object());
            }
        }
        let carried: Vec<&ReleaseIntent> = markers
            .iter()
            .filter(|marker| {
                destination_marker_kind(marker) == Some(DestinationMarkerKind::Release)
            })
            .filter_map(|marker| {
                self.releases
                    .iter()
                    .find(|release| release.marker() == Some(*marker))
            })
            .collect();
        let contradicted: Vec<&FastContradictionIntent> = owners
            .iter()
            .filter(|owner| fast_marker_kind(owner) == Some(FastMarkerKind::Contradiction))
            .filter_map(|marker| self.contradictions.iter().find(|c| c.marker() == Some(*marker)))
            .collect();
        let shared = match (carried.first(), contradicted.first()) {
            (Some(first), _) => Some((
                first.finality.clone(),
                first.base.clone(),
                first.account_proof.clone(),
                first.attestation_object(),
            )),
            (None, Some(first)) => Some((
                first.finality.clone(),
                Some(first.base.clone()),
                first.account_proof.clone(),
                None,
            )),
            (None, None) => None,
        };
        if let Some((finality, base, account_proof, attestation)) = shared {
            objects.push(DestinationObject {
                kind: DestinationObjectKind::FinalityUpdate,
                payload: finality,
            });
            if let Some(base) = base {
                objects.push(DestinationObject {
                    kind: DestinationObjectKind::BaseGameProof,
                    payload: base.game_proof,
                });
                objects.push(DestinationObject {
                    kind: DestinationObjectKind::OutputRootPreimage,
                    payload: base.output_preimage,
                });
            }
            objects.push(DestinationObject {
                kind: DestinationObjectKind::BridgeAccountProof,
                payload: account_proof,
            });
            objects.extend(attestation);
            objects.extend(carried.iter().map(|release| release.release.object()));
            objects.extend(contradicted.iter().map(|c| c.object()));
        }
        if let Some(revocation) = &self.revocation {
            let revoked = jetsam_chain::consensus::destination::fast::FastRevocation::decode_payload(
                &revocation.payload,
            )
            .map(|r| jetsam_chain::consensus::destination::fast::revocation_marker(&r.key));
            if revoked.is_ok_and(|marker| owners.contains(&marker)) {
                objects.push(revocation.object());
            }
        }
        DestinationSection { objects }
    }

    /// The offer without what `culprits` blames.
    pub fn without(&self, culprits: &Culprits) -> Self {
        Self {
            committee_update: self
                .committee_update
                .clone()
                .filter(|_| !culprits.committee_update),
            releases: self
                .releases
                .iter()
                .filter(|release| !culprits.releases.contains(&release.release.n))
                .cloned()
                .collect(),
            contradictions: self
                .contradictions
                .iter()
                .filter(|c| !culprits.contradictions.contains(&c.n()))
                .cloned()
                .collect(),
            revocation: self.revocation.clone().filter(|_| !culprits.revocation),
        }
    }

    fn fits(&self, budget: usize) -> bool {
        self.object_count() <= budget
            && destination_section_wire_len(&self.section_for(&self.payment_order()))
                <= MAX_DESTINATION_SECTION_BYTES
    }
}

fn anchor_of(payment: &PagedSpendIntent) -> Option<[u8; 32]> {
    payment.pages.first().map(|page| page.body.epoch_anchor)
}

/// What the block at `height` after a branch in `state` offers, from the
/// intents the mempool holds, with `other_objects` already in its section
/// (module docs). `anchor_ok` says whether a transaction's epoch anchor is
/// minable at `height`.
pub fn plan_destination_offer(
    state: &DestinationState,
    rules: &DestinationRules,
    height: u64,
    committee_updates: Vec<CommitteeUpdateIntent>,
    releases: Vec<ReleaseIntent>,
    anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    other_objects: usize,
) -> DestinationOffer {
    plan_destination_offer_with(
        state,
        rules,
        height,
        committee_updates,
        releases,
        Vec::new(),
        None,
        anchor_ok,
        other_objects,
    )
}

/// [`plan_destination_offer`] with the fast lane's intents (from the
/// return-over-Base height on): the revocation if its key is still open (it
/// may share a block with a committee update); contradictions of open fast
/// records, never beside a committee update nor a fast release, under the
/// Base objects of the releases offered (or of the first contradiction).
#[allow(clippy::too_many_arguments)]
pub fn plan_destination_offer_with(
    state: &DestinationState,
    rules: &DestinationRules,
    height: u64,
    committee_updates: Vec<CommitteeUpdateIntent>,
    releases: Vec<ReleaseIntent>,
    contradictions: Vec<FastContradictionIntent>,
    revocation: Option<FastRevocationIntent>,
    anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    other_objects: usize,
) -> DestinationOffer {
    let mut offer = plan_releases(state, rules, height, committee_updates, releases, anchor_ok, other_objects);
    let budget = MAX_BLOCK_DESTINATION_OBJECTS.saturating_sub(other_objects);
    if !rules.base_return.active_at(height) || rules.bridge.is_none() {
        return offer;
    }
    if let Some(revocation) = revocation {
        let open = jetsam_chain::consensus::destination::fast::FastRevocation::decode_payload(
            &revocation.payload,
        )
        .is_ok_and(|r| !state.fast.is_closed(&r.key));
        if open && anchor_of(&revocation.payment).is_some_and(|a| anchor_ok(&a)) {
            let mut candidate = offer.clone();
            candidate.revocation = Some(revocation);
            if candidate.fits(budget) {
                offer = candidate;
            }
        }
    }
    if offer.committee_update.is_some()
        || offer.releases.first().is_some_and(|first| first.fast.is_some())
    {
        return offer;
    }
    for contradiction in contradictions {
        let n = contradiction.n();
        let open = state.fast.records.get(&n).is_some_and(|r| !r.contradicted);
        if !open || !anchor_of(&contradiction.payment).is_some_and(|a| anchor_ok(&a)) {
            continue;
        }
        let shares = match (offer.releases.first(), offer.contradictions.first()) {
            (Some(first), _) => {
                first.finality == contradiction.finality
                    && first.account_proof == contradiction.account_proof
                    && first.base.as_ref() == Some(&contradiction.base)
            }
            (None, Some(first)) => {
                first.finality == contradiction.finality
                    && first.account_proof == contradiction.account_proof
                    && first.base == contradiction.base
            }
            (None, None) => true,
        };
        if !shares {
            continue;
        }
        let mut candidate = offer.clone();
        candidate.contradictions.push(contradiction);
        if candidate.fits(budget) {
            offer = candidate;
        }
    }
    offer
}

/// The committee update and the releases of [`plan_destination_offer`].
fn plan_releases(
    state: &DestinationState,
    rules: &DestinationRules,
    height: u64,
    committee_updates: Vec<CommitteeUpdateIntent>,
    mut releases: Vec<ReleaseIntent>,
    anchor_ok: &dyn Fn(&[u8; 32]) -> bool,
    other_objects: usize,
) -> DestinationOffer {
    let budget = MAX_BLOCK_DESTINATION_OBJECTS.saturating_sub(other_objects);
    let mut offer = DestinationOffer::default();
    if let Some(anchor) = rules.anchor {
        let due = state
            .committees
            .last()
            .map_or(Some(anchor.period), |known| known.period.checked_add(1));
        offer.committee_update = committee_updates.into_iter().find(|update| {
            due.is_some()
                && update.next_period() == due
                && anchor_of(&update.payment).is_some_and(|a| anchor_ok(&a))
        });
        if !offer.fits(budget) {
            offer.committee_update = None;
        }
    }
    if rules.bridge.is_none() {
        return offer;
    }
    let over_base = rules.base_return.active_at(height);
    // D107 §9.1: never a game proof beside a committee update.
    if over_base && offer.committee_update.is_some() {
        return offer;
    }
    releases.sort_by_key(|release| release.release.n);
    let mut window = state.released_in_window(height, rules.vault.window_blocks);
    for release in releases {
        if offer.releases.len() >= rules.vault.max_releases_per_block {
            break;
        }
        if state.released.contains(release.release.n)
            || !anchor_of(&release.payment).is_some_and(|a| anchor_ok(&a))
        {
            continue;
        }
        // Over Base only from its height on, paying what V2 asks there.
        if over_base != release.base.is_some()
            || (over_base
                && release.paid_to()
                    != Some(release_payee(
                        &state.advances,
                        release.release.n,
                        &release.release.owner,
                        release.release.amount,
                    )))
        {
            continue;
        }
        if let Some(first) = offer.releases.first() {
            if !release.shares_proofs_with(first) {
                continue;
            }
        }
        // Never jump a payable release that waits only for the window (D123:
        // the window cap of the block at `height`).
        let Some(after) = window
            .checked_add(release.release.amount)
            .filter(|after| *after <= rules.vault.at(height).window_cap)
        else {
            break;
        };
        let mut candidate = offer.clone();
        candidate.releases.push(release);
        if !candidate.fits(budget) {
            break;
        }
        window = after;
        offer = candidate;
    }
    offer
}

/// The entries of `entries` (the mempool's selection, `pages_of` their pages)
/// that keep V8 when the chain template places them with `offer`'s
/// transactions by decreasing fee then logical txid, the vault holding
/// `bridge_locked` before the block at `height`: a deposit that would take it
/// over that block's ceiling (D123: [`VaultRules::at`]) is left out (it stays
/// in the pool), a release gives back what it takes. Order of `entries` kept.
pub fn deposits_within_ceiling<E>(
    entries: Vec<E>,
    pages_of: impl Fn(&E) -> &[TxPage],
    offer: &DestinationOffer,
    bridge_locked: u64,
    rules: &VaultRules,
    height: u64,
) -> Vec<E> {
    // D123: the ceiling of the block at `height`.
    let rules = &rules.at(height);
    enum Effect {
        Release(u64),
        Deposit(usize, u64),
        Other,
    }
    let vault_outputs = |pages: &[TxPage]| -> (bool, u64) {
        let mut deposit = false;
        let mut to_vault = 0u64;
        for page in pages {
            for (_, output) in page.body.live_outputs() {
                if destination_marker_kind(&output.owner) == Some(DestinationMarkerKind::Deposit) {
                    deposit = true;
                }
                if output.owner == rules.vault {
                    to_vault = to_vault.saturating_add(output.amount);
                }
            }
        }
        (deposit, to_vault)
    };
    let mut placed: Vec<(u64, [u8; 32], Effect)> = Vec::new();
    for release in &offer.releases {
        if let Ok(facts) = jetsam_tx::validate_paged_spend(&release.payment.pages) {
            let (_, change) = vault_outputs(&release.payment.pages);
            let taken = u64::try_from(facts.input_sum).unwrap_or(u64::MAX);
            placed.push((
                facts.fee,
                facts.logical_txid.0,
                Effect::Release(taken.saturating_sub(change)),
            ));
        }
    }
    for (index, entry) in entries.iter().enumerate() {
        let pages = pages_of(entry);
        let Ok(facts) = jetsam_tx::validate_paged_spend(pages) else {
            continue;
        };
        let (deposit, amount) = vault_outputs(pages);
        let effect = if deposit {
            Effect::Deposit(index, amount)
        } else {
            Effect::Other
        };
        placed.push((facts.fee, facts.logical_txid.0, effect));
    }
    placed.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let mut locked = bridge_locked;
    let mut dropped = std::collections::HashSet::new();
    for (_, _, effect) in placed {
        match effect {
            Effect::Release(outflow) => locked = locked.saturating_sub(outflow),
            Effect::Deposit(index, amount) => match locked.checked_add(amount) {
                Some(after) if after <= rules.outstanding_ceil => locked = after,
                _ => {
                    dropped.insert(index);
                }
            },
            Effect::Other => {}
        }
    }
    entries
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !dropped.contains(index))
        .map(|(_, entry)| entry)
        .collect()
}

/// The entries of `entries` (the mempool's selection, `pages_of` their pages)
/// whose advance and claim-transfer groups the block at `height` after a
/// branch in `state` accepts together with `offer`'s releases (the return over
/// Base, D107, D114): one by one in the order the chain template places them
/// (decreasing fee, then logical txid), an entry the rule refuses beside
/// those kept before it is left out (it stays in the pool). The rule refuses
/// only what another group brings (a burn twice, a cap, a transfer twice, the
/// release of its `n`) or what the branch holds, so the entries kept are
/// accepted in any order, and without the offer too. Every entry below the
/// return-over-Base height (none today). Order of `entries` kept.
pub fn advances_within_rules<E>(
    entries: Vec<E>,
    pages_of: impl Fn(&E) -> &[TxPage],
    offer: &DestinationOffer,
    state: &DestinationState,
    rules: &DestinationRules,
    height: u64,
) -> Vec<E> {
    if !rules.base_return.active_at(height) {
        return entries;
    }
    let released_here: Vec<u64> = offer
        .releases
        .iter()
        .map(|release| release.release.n)
        .collect();
    let mut check = AdvanceCheck::new(
        &state.advances,
        &state.released,
        released_here,
        &rules.vault,
        &rules.base_return.advance,
    );
    let mut placed: Vec<(u64, [u8; 32], usize)> = entries
        .iter()
        .enumerate()
        .filter_map(|(index, entry)| {
            jetsam_tx::validate_paged_spend(pages_of(entry))
                .ok()
                .map(|facts| (facts.fee, facts.logical_txid.0, index))
        })
        .collect();
    placed.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
    let mut dropped = std::collections::HashSet::new();
    for (_, _, index) in placed {
        let pages = pages_of(&entries[index]);
        let outputs: Vec<(Address, u64)> = pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
            .collect();
        let from = pages
            .first()
            .map_or(Address([0; 32]), |page| page.body.input_owner);
        if check.group(index, from, &outputs).is_err() {
            dropped.insert(index);
        }
    }
    entries
        .into_iter()
        .enumerate()
        .filter(|(index, _)| !dropped.contains(index))
        .map(|(_, entry)| entry)
        .collect()
}

/// What a refused block blames among the destination objects it carried.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Culprits {
    /// Withdrawal indices to forget.
    pub releases: Vec<u64>,
    /// Whether to forget the committee update.
    pub committee_update: bool,
    /// The fast lane: the contradictions (by `n`) to forget.
    pub contradictions: Vec<u64>,
    /// The fast lane: whether to forget the revocation.
    pub revocation: bool,
}

/// The objects `error` (from `validate_block_destination` on a block carrying
/// `offer`) blames: one release for its own proof or an index already paid;
/// every release for their shared finality update or account proof; the
/// committee update for the anchor, its payload or the light client when it
/// was carried (else the releases' finality update); the committee update
/// for a vault-gate fault of one group (A2-M2: an output to the vault, a
/// deposit, a marker with value, a spend from the vault) when its own payment
/// has that fault ([`check_object_payment_vault`], `rules`) — the group index
/// is the block's, the payment is found by what it carries. Nothing for a
/// local fault (keys this node lacks, a state not the parent's) or for what
/// the template itself decided (counts, window): the block is then built
/// without destination objects and they stay held.
pub fn culprits_of(
    error: &DestinationBlockError,
    offer: &DestinationOffer,
    rules: &VaultRules,
) -> Culprits {
    let all_releases = || {
        offer
            .releases
            .iter()
            .map(|release| release.release.n)
            .collect()
    };
    let one = |n: u64| Culprits {
        releases: offer
            .releases
            .iter()
            .map(|release| release.release.n)
            .filter(|offered| *offered == n)
            .collect(),
        committee_update: false,
        ..Culprits::default()
    };
    let update = || Culprits {
        releases: Vec::new(),
        committee_update: offer.committee_update.is_some(),
        ..Culprits::default()
    };
    let contradiction = |n: u64| Culprits {
        contradictions: vec![n],
        ..Culprits::default()
    };
    // The fast lane's key or statement: the fast releases and the revocation.
    let fast_lane = || Culprits {
        releases: if offer.releases.first().is_some_and(|r| r.fast.is_some()) {
            all_releases()
        } else {
            Vec::new()
        },
        revocation: offer.revocation.is_some(),
        ..Culprits::default()
    };
    match error {
        DestinationBlockError::NoFastRecord { n }
        | DestinationBlockError::AlreadyContradicted { n }
        | DestinationBlockError::ContradictionTooEarly { n }
        | DestinationBlockError::NotAContradiction { n } => contradiction(*n),
        DestinationBlockError::FastKeyUnset
        | DestinationBlockError::FastRevokeKeyUnset
        | DestinationBlockError::FastKeyClosed
        | DestinationBlockError::FastBadSignature
        | DestinationBlockError::FastStatementMismatch
        | DestinationBlockError::FastStatementNotTheGame
        | DestinationBlockError::MalformedObject {
            kind:
                DestinationObjectKind::FastGameAttestation | DestinationObjectKind::FastKeyRevocation,
        } => fast_lane(),
        DestinationBlockError::FastObjectsDoNotMatchMarkers
        | DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::FastContradiction,
        } => Culprits {
            contradictions: offer.contradictions.iter().map(|c| c.n()).collect(),
            revocation: offer.revocation.is_some(),
            ..Culprits::default()
        },
        DestinationBlockError::Vault(VaultError::NotProvenOnL1 { n, .. })
        | DestinationBlockError::Vault(VaultError::ReleaseAlreadyPaid { n }) => one(*n),
        DestinationBlockError::LightClient(_) if offer.committee_update.is_some() => update(),
        DestinationBlockError::LightClient(_)
        | DestinationBlockError::BridgeAccount(_)
        | DestinationBlockError::BaseReturn(_)
        | DestinationBlockError::L1TimeUnknown
        | DestinationBlockError::MalformedObject {
            kind:
                DestinationObjectKind::FinalityUpdate
                | DestinationObjectKind::BridgeAccountProof
                | DestinationObjectKind::VaultRelease
                | DestinationObjectKind::BaseGameProof
                | DestinationObjectKind::OutputRootPreimage,
        } => Culprits {
            releases: all_releases(),
            committee_update: false,
            ..Culprits::default()
        },
        DestinationBlockError::AnchorMismatch
        | DestinationBlockError::MalformedObject {
            kind: DestinationObjectKind::CommitteeUpdate,
        } => update(),
        DestinationBlockError::Vault(
            VaultError::VaultOutputOutsideDeposit { .. }
            | VaultError::DepositShape { .. }
            | VaultError::DepositAmount { .. }
            | VaultError::DepositBeforeBridgeOpening { .. }
            | VaultError::VaultCeilingReached { .. }
            | VaultError::MarkerCarriesValue { .. }
            | VaultError::SpendFromLockedAddress { .. }
            | VaultError::ReleaseMarkerOutsideRelease { .. },
        ) => Culprits {
            releases: Vec::new(),
            committee_update: offer.committee_update.as_ref().is_some_and(|update| {
                check_object_payment_vault(&update.payment.pages, rules).is_err()
            }),
            ..Culprits::default()
        },
        _ => Culprits::default(),
    }
}

/// A2-M1: what a template does after the block carrying `offer` was refused
/// with `error` by the destination rules, before it gives up the whole
/// offer: it tries the offer without what the refusal blames ([`culprits_of`],
/// [`DestinationOffer::without`]) — and again without what a new refusal of
/// that one blames, while something is blamed and something is left. So one
/// bad release (or a bad committee update) no longer takes the others out of
/// the block. `attempt` builds and judges the block carrying an offer: `Ok`,
/// or `Err(Some(error))` for a destination refusal, `Err(None)` for any
/// other (no further try). Returns the first accepted build, if any, and
/// everything blamed on the way (for the pool to forget).
pub fn retry_without_culprits<T>(
    offer: &DestinationOffer,
    error: &DestinationBlockError,
    rules: &VaultRules,
    mut attempt: impl FnMut(&DestinationOffer) -> Result<T, Option<DestinationBlockError>>,
) -> (Option<T>, Culprits) {
    let mut blamed = Culprits::default();
    let mut current = offer.clone();
    let mut culprits = culprits_of(error, &current, rules);
    loop {
        let reduced = current.without(&culprits);
        if reduced.is_empty() || reduced == current {
            blamed.add(&culprits);
            return (None, blamed);
        }
        blamed.add(&culprits);
        match attempt(&reduced) {
            Ok(built) => return (Some(built), blamed),
            Err(Some(error)) => {
                culprits = culprits_of(&error, &reduced, rules);
                current = reduced;
            }
            Err(None) => return (None, blamed),
        }
    }
}

impl Culprits {
    /// Blame `other`'s objects too.
    fn add(&mut self, other: &Culprits) {
        for n in &other.releases {
            if !self.releases.contains(n) {
                self.releases.push(*n);
            }
        }
        self.committee_update |= other.committee_update;
        for n in &other.contradictions {
            if !self.contradictions.contains(n) {
                self.contradictions.push(*n);
            }
        }
        self.revocation |= other.revocation;
    }
}

#[cfg(test)]
mod tests;
