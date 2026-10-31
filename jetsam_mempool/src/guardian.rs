// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T3.7): the guardian objects this node holds for its miners.
//!
//! A guardian object (attestation 0x20, equivocation 0x21, and the dormant
//! delegation's 0x22-0x26) never travels alone, and never as a plain
//! transaction: it travels as an **intent** with the transaction that pays
//! for it and carries its marker (T1.1 §10). Alone, that transaction is
//! refused, as is any plain transaction that spends from, or pays, the stake
//! vault or the reward pool.
//!
//! Admission judges an intent as the next block would judge a block of that
//! one group after the objects already held (the conflicts of §10 follow:
//! only the highest attestation, one equivocation, an unlock beyond the
//! stake or a second payout of an entry refused), on the guardian state of
//! the tip, with two exceptions: F-4 is the block's, so a reward funding
//! before the latch waits for its pair; and the BLS signatures come last,
//! after every cheap refusal, behind a limit of
//! [`GUARDIAN_RATE_LIMIT_PER_SECOND`] attestations and equivocations per
//! second and per source (A1 m13).
//!
//! The held intents are judged again at every new tip (period boundaries,
//! `J_D` and the latch included), and forgotten once refused or once their
//! inputs are spent.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use jetsam_chain::consensus::client_objects::{client_object_marker_kind_at, ClientBatchRules};
use jetsam_chain::consensus::guardian::{
    apply_equivocation, apply_payout, apply_period_boundary, apply_reward_claim, apply_reward_fund,
    apply_stake_lock, apply_stake_unlock, check_attestation, check_equivocation,
    check_group_pool_gate, guardian_body_len_allowed, guardian_marker_kind, GuardianAttestation,
    GuardianBranch, GuardianEquivocation, GuardianError, GuardianGroupFacts, GuardianObject,
    GuardianObjectKind, GuardianPayout, GuardianRules, GuardianState, GuardianStatement,
    RewardClaim, RewardFund, StakeLock, StakeUnlock, GROUP_KEY_BYTES,
    MAX_DELEGATION_OBJECTS_PER_BLOCK,
};
use jetsam_chain::consensus::params::v1_7_active_with;
use jetsam_chain::consensus::vault::destination_marker_kind;
use jetsam_chain::consensus::wire_limits::MAX_TX_INTENT_BYTES_GLOBAL;
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{PagedSpendIntent, TxPage};

/// First bytes of an encoded guardian intent.
pub const GUARDIAN_INTENT_MAGIC: [u8; 8] = *b"JTMGIN01";
/// Largest guardian body (an equivocation with two anchor-leaf witnesses).
pub const MAX_GUARDIAN_BODY_BYTES: usize = 3_227;
/// Largest encoded intent: magic, kind, the body and the payment, each with
/// its length.
pub const MAX_GUARDIAN_INTENT_BYTES: usize =
    GUARDIAN_INTENT_MAGIC.len() + 1 + 4 + MAX_GUARDIAN_BODY_BYTES + 4 + MAX_TX_INTENT_BYTES_GLOBAL;
/// Delegation objects held at once: what one block can carry.
pub const MAX_HELD_DELEGATION_OBJECTS: usize = MAX_DELEGATION_OBJECTS_PER_BLOCK;
/// Attestations and equivocations one source may hand in per second
/// (A1 m13); beyond, refused without any verification.
pub const GUARDIAN_RATE_LIMIT_PER_SECOND: usize = 4;
/// Sources the limiter remembers at once (beyond, a new source is refused
/// until older ones fall out of their window).
pub const MAX_RATE_LIMITED_SOURCES: usize = 4_096;
/// A3-5: the source of every RPC client on this node's own host. Its quota
/// is its own and is always kept, whatever the number of other sources.
pub const LOCAL_RPC_SOURCE: &str = "rpc:local";

/// One guardian object with its transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardianIntent {
    pub object: GuardianObject,
    /// Carries the object's marker (value 0).
    pub payment: PagedSpendIntent,
}

/// Why bytes are not a guardian intent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuardianIntentError {
    TooLarge {
        actual: usize,
        max: usize,
    },
    BadMagic,
    UnknownKind(u8),
    Truncated,
    TrailingBytes,
    /// A body of a length its kind never has.
    BodyLength {
        kind: GuardianObjectKind,
        len: usize,
    },
    /// A body that does not decode as its object.
    MalformedObject(GuardianError),
    /// The transaction does not decode, or is not this object's.
    Payment(String),
}

impl std::fmt::Display for GuardianIntentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

fn take<'a>(src: &mut &'a [u8]) -> Result<&'a [u8], GuardianIntentError> {
    if src.len() < 4 {
        return Err(GuardianIntentError::Truncated);
    }
    let len = u32::from_le_bytes(src[..4].try_into().expect("four bytes")) as usize;
    let rest = &src[4..];
    if rest.len() < len {
        return Err(GuardianIntentError::Truncated);
    }
    let (head, tail) = rest.split_at(len);
    *src = tail;
    Ok(head)
}

/// Whether `body` decodes as a `kind` object (lengths, the claim's account
/// byte, an equivocation's rule and witnesses).
fn decode_object(kind: GuardianObjectKind, body: &[u8]) -> Result<(), GuardianError> {
    match kind {
        GuardianObjectKind::Attestation => GuardianAttestation::decode(body).map(drop),
        GuardianObjectKind::Equivocation => GuardianEquivocation::decode(body).map(drop),
        GuardianObjectKind::StakeLock => StakeLock::decode(body).map(drop),
        GuardianObjectKind::StakeUnlock => StakeUnlock::decode(body).map(drop),
        GuardianObjectKind::RewardFund => RewardFund::decode(body).map(drop),
        GuardianObjectKind::RewardClaim => RewardClaim::decode(body).map(drop),
        GuardianObjectKind::Payout => GuardianPayout::decode(body).map(drop),
    }
}

impl GuardianIntent {
    /// `magic ‖ kind (section code) ‖ len:u32 LE ‖ body ‖ len:u32 LE ‖
    /// payment`.
    pub fn encode(&self) -> Result<Vec<u8>, GuardianIntentError> {
        let payment = self
            .payment
            .to_bytes()
            .map_err(|e| GuardianIntentError::Payment(format!("{e:?}")))?;
        let mut out = GUARDIAN_INTENT_MAGIC.to_vec();
        out.push(self.object.kind.section_code());
        out.extend_from_slice(&(self.object.body.len() as u32).to_le_bytes());
        out.extend_from_slice(&self.object.body);
        out.extend_from_slice(&(payment.len() as u32).to_le_bytes());
        out.extend_from_slice(&payment);
        Ok(out)
    }

    /// The canonical decoding, bounded before anything is allocated beyond
    /// the input: the body decodes as its object, and the transaction carries
    /// exactly one guardian marker, this object's.
    pub fn decode(bytes: &[u8]) -> Result<Self, GuardianIntentError> {
        if bytes.len() > MAX_GUARDIAN_INTENT_BYTES {
            return Err(GuardianIntentError::TooLarge {
                actual: bytes.len(),
                max: MAX_GUARDIAN_INTENT_BYTES,
            });
        }
        let Some(rest) = bytes.strip_prefix(&GUARDIAN_INTENT_MAGIC) else {
            return Err(GuardianIntentError::BadMagic);
        };
        let (&code, mut src) = rest.split_first().ok_or(GuardianIntentError::Truncated)?;
        let kind = GuardianObjectKind::from_section_code(code)
            .ok_or(GuardianIntentError::UnknownKind(code))?;
        let body = take(&mut src)?;
        if !guardian_body_len_allowed(kind, body.len()) {
            return Err(GuardianIntentError::BodyLength {
                kind,
                len: body.len(),
            });
        }
        decode_object(kind, body).map_err(GuardianIntentError::MalformedObject)?;
        let payment_bytes = take(&mut src)?;
        if !src.is_empty() {
            return Err(GuardianIntentError::TrailingBytes);
        }
        if payment_bytes.len() > MAX_TX_INTENT_BYTES_GLOBAL {
            return Err(GuardianIntentError::TooLarge {
                actual: payment_bytes.len(),
                max: MAX_TX_INTENT_BYTES_GLOBAL,
            });
        }
        let mut payment_src = payment_bytes;
        let payment = PagedSpendIntent::decode(&mut payment_src)
            .map_err(|e| GuardianIntentError::Payment(format!("{e:?}")))?;
        if !payment_src.is_empty() {
            return Err(GuardianIntentError::TrailingBytes);
        }
        let intent = Self {
            object: GuardianObject::new(kind, body.to_vec()),
            payment,
        };
        let markers: Vec<Address> = intent
            .payment
            .pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| o.owner))
            .filter(|owner| guardian_marker_kind(owner).is_some())
            .collect();
        if markers != [intent.object.marker()] {
            return Err(GuardianIntentError::Payment(
                "the transaction does not carry exactly this object's marker".into(),
            ));
        }
        Ok(intent)
    }

    /// The height an attestation describes (`None` for another kind, or a
    /// statement that does not decode).
    pub fn attested_height(&self) -> Option<u64> {
        if self.object.kind != GuardianObjectKind::Attestation {
            return None;
        }
        let attestation = GuardianAttestation::decode(&self.object.body).ok()?;
        GuardianStatement::decode(&attestation.statement)
            .ok()
            .map(|statement| statement.height)
    }
}

/// The facts of the one group a payment makes (index 0), as a block's
/// guardian rules read them.
pub fn group_facts(pages: &[TxPage], input_owner: Address, fee: u64) -> GuardianGroupFacts {
    GuardianGroupFacts {
        group: 0,
        input_owner,
        outputs: pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, o)| (o.owner, o.amount)))
            .collect(),
        fee,
        input_creation_ids: pages
            .iter()
            .flat_map(|page| page.body.live_inputs().map(|(_, i)| i.creation_id))
            .collect(),
    }
}

/// M-2, M-3 and M-5 for one group at `height` (as `guardian_context` judges
/// each group of a block).
pub fn check_group_markers(group: &GuardianGroupFacts, height: u64) -> Result<(), GuardianError> {
    if guardian_marker_kind(&group.input_owner).is_some() {
        return Err(GuardianError::SpendFromLockedAddress { group: group.group });
    }
    let batch = ClientBatchRules::current();
    let mut markers = 0usize;
    for (owner, amount) in &group.outputs {
        if guardian_marker_kind(owner).is_some() {
            if *amount != 0 {
                return Err(GuardianError::GuardianMarkerCarriesValue { group: group.group });
            }
            markers += 1;
        } else if client_object_marker_kind_at(owner, height, &batch).is_some()
            || destination_marker_kind(owner).is_some()
        {
            markers += 1;
        }
    }
    if markers > 1 {
        return Err(GuardianError::MultipleMarkersInTransaction { group: group.group });
    }
    Ok(())
}

/// The policy for a transaction that does not travel with a guardian object
/// (T1.1 §10), for the block at `next_height`: no spend from either pool or
/// from a guardian marker, no output to either pool, no guardian marker
/// (a marker without its object, the design's `GuardianPaymentWithoutObject`:
/// M-6 in a block). Inert below J.
pub fn check_plain_guardian_transaction(
    pages: &[TxPage],
    next_height: u64,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    if !v1_7_active_with(next_height, rules.activation_height) {
        return Ok(());
    }
    for page in pages {
        let owner = page.body.input_owner;
        if owner == rules.stake_vault || owner == rules.reward_pool {
            return Err(GuardianError::SpendFromGuardianPoolWithoutPayout { group: 0 });
        }
        if guardian_marker_kind(&owner).is_some() {
            return Err(GuardianError::SpendFromLockedAddress { group: 0 });
        }
        for (_, output) in page.body.live_outputs() {
            if output.owner == rules.stake_vault || output.owner == rules.reward_pool {
                return Err(GuardianError::OutputToGuardianPoolOutsideObject { group: Some(0) });
            }
            if guardian_marker_kind(&output.owner).is_some() {
                return Err(GuardianError::ObjectsDoNotMatchMarkers);
            }
        }
    }
    Ok(())
}

/// How [`apply_guardian_intent`] treats the guardians' BLS signatures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signatures {
    /// Verified with every other rule (the consensus order: last).
    Check,
    /// Not verified: every other rule is, and the signatures are verified
    /// apart (admission, outside the pool's lock) or were already verified
    /// (a held object judged again).
    Defer,
}

/// `rules` with a group key no signature can verify under: the consensus
/// checks then run every rule, and fail at the signature on the key itself,
/// at once, without any pairing. A missing key stays missing (A-2, Q-1).
pub fn unusable_key_rules(rules: &GuardianRules) -> GuardianRules {
    GuardianRules {
        group_key: rules.group_key.map(|_| [0u8; GROUP_KEY_BYTES]),
        ..*rules
    }
}

/// The guardian rules of one group carrying `object` (markers, the pools'
/// gate, the object's rules and effect, as `apply_block_guardian` applies
/// each object of a block), applied to `state` for the block at `height`.
/// The state is unchanged on a refusal. The period boundary is the caller's
/// (once per block, [`project`]); F-4 too.
pub fn apply_guardian_intent(
    state: &mut GuardianState,
    height: u64,
    group: &GuardianGroupFacts,
    object: &GuardianObject,
    branch: &dyn GuardianBranch,
    rules: &GuardianRules,
    signatures: Signatures,
) -> Result<(), GuardianError> {
    check_group_markers(group, height)?;
    check_group_pool_gate(group, Some(object.kind), rules)?;
    let deferred = signatures == Signatures::Defer && rules.group_key.is_some();
    let judged = if deferred {
        unusable_key_rules(rules)
    } else {
        *rules
    };
    let unsigned =
        |error: GuardianError| deferred && error == GuardianError::InvalidGuardianSignature;
    match object.kind {
        GuardianObjectKind::Attestation => {
            let attestation = GuardianAttestation::decode(&object.body)?;
            let attested = match check_attestation(state, height, &attestation, branch, &judged) {
                Ok(attested) => attested,
                Err(error) if unsigned(error.clone()) => {
                    GuardianStatement::decode(&attestation.statement)?.height
                }
                Err(error) => return Err(error),
            };
            state.last_attested_height = Some(attested);
            state.attested_period = Some(height / rules.period);
        }
        GuardianObjectKind::Equivocation => {
            let equivocation = GuardianEquivocation::decode(&object.body)?;
            match check_equivocation(state, &equivocation, &judged) {
                Ok(()) => {}
                Err(error) if unsigned(error.clone()) => {}
                Err(error) => return Err(error),
            }
            apply_equivocation(state, height);
        }
        GuardianObjectKind::StakeLock => apply_stake_lock(
            state,
            height,
            group,
            &StakeLock::decode(&object.body)?,
            rules,
        )?,
        GuardianObjectKind::StakeUnlock => apply_stake_unlock(
            state,
            height,
            group,
            &StakeUnlock::decode(&object.body)?,
            rules,
        )?,
        GuardianObjectKind::Payout => apply_payout(
            state,
            height,
            group,
            &GuardianPayout::decode(&object.body)?,
            rules,
        )?,
        GuardianObjectKind::RewardFund => apply_reward_fund(
            state,
            height,
            group,
            &RewardFund::decode(&object.body)?,
            rules,
        )?,
        GuardianObjectKind::RewardClaim => apply_reward_claim(
            state,
            height,
            group,
            &RewardClaim::decode(&object.body)?,
            rules,
        )?,
    }
    Ok(())
}

/// Verify the guardians' signature(s) of an attestation or an equivocation
/// under `rules` (the costly rule, A-10 / Q-5); nothing for another kind.
pub fn verify_guardian_signatures(
    object: &GuardianObject,
    rules: &GuardianRules,
) -> Result<(), GuardianError> {
    let bad = |_| GuardianError::InvalidGuardianSignature;
    match object.kind {
        GuardianObjectKind::Attestation => {
            let attestation = GuardianAttestation::decode(&object.body)?;
            rules
                .verify_signature(&attestation.statement, &attestation.signature)
                .map_err(bad)
        }
        GuardianObjectKind::Equivocation => {
            let equivocation = GuardianEquivocation::decode(&object.body)?;
            rules
                .verify_signature(&equivocation.s1, &equivocation.sig1)
                .and_then(|()| rules.verify_signature(&equivocation.s2, &equivocation.sig2))
                .map_err(bad)
        }
        _ => Ok(()),
    }
}

/// An intent the pool holds: its arrival number, the intent, the facts of
/// its group.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeldGuardian {
    pub seq: u64,
    pub intent: GuardianIntent,
    pub group: GuardianGroupFacts,
}

/// The state after the block at `height` carried `held` (in that order) on
/// a branch whose state is `state`: the period boundary once, then each
/// object; the ones refused leave the state as it was and are returned with
/// their refusal. Signatures are not verified again ([`Signatures::Defer`]).
pub fn project<'a>(
    state: &GuardianState,
    height: u64,
    held: impl IntoIterator<Item = &'a HeldGuardian>,
    branch: &dyn GuardianBranch,
    rules: &GuardianRules,
) -> (GuardianState, Vec<(u64, GuardianError)>) {
    let mut next = state.clone();
    if height.is_multiple_of(rules.period) {
        apply_period_boundary(&mut next, height, rules);
    }
    let mut refused = Vec::new();
    for held in held {
        if let Err(error) = apply_guardian_intent(
            &mut next,
            height,
            &held.group,
            &held.intent.object,
            branch,
            rules,
            Signatures::Defer,
        ) {
            refused.push((held.seq, error));
        }
    }
    (next, refused)
}

/// Remembers, per source, when it last handed in attestations and
/// equivocations.
#[derive(Clone, Debug, Default)]
pub struct GuardianRateLimiter {
    seen: HashMap<String, VecDeque<Instant>>,
}

impl GuardianRateLimiter {
    /// Whether `source` may hand in one more at `now`: fewer than
    /// [`GUARDIAN_RATE_LIMIT_PER_SECOND`] in the last second. A refusal is not
    /// counted. [`LOCAL_RPC_SOURCE`] is remembered even when
    /// [`MAX_RATE_LIMITED_SOURCES`] others are (A3-5).
    pub fn admit(&mut self, source: &str, now: Instant) -> bool {
        let window = Duration::from_secs(1);
        let fresh = |time: &Instant| now.saturating_duration_since(*time) < window;
        // A3-5: the local host's quota is always kept.
        if source != LOCAL_RPC_SOURCE
            && !self.seen.contains_key(source)
            && self.seen.len() >= MAX_RATE_LIMITED_SOURCES
        {
            self.seen.retain(|_, times| times.back().is_some_and(fresh));
            if self.seen.len() >= MAX_RATE_LIMITED_SOURCES {
                return false;
            }
        }
        let times = self.seen.entry(source.to_string()).or_default();
        while times.front().is_some_and(|time| !fresh(time)) {
            times.pop_front();
        }
        if times.len() >= GUARDIAN_RATE_LIMIT_PER_SECOND {
            return false;
        }
        times.push_back(now);
        true
    }
}

/// Why the pool does not hold an intent (cheap, before any rule).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldRefusal {
    /// The same object is already held.
    AlreadyHeld,
    /// An attestation of a height at most that of the one held.
    Covered { held: u64 },
    /// An equivocation is already held (one per branch is ever applied).
    EquivocationHeld,
    /// [`MAX_HELD_DELEGATION_OBJECTS`] delegation objects already held.
    Full,
    /// Its transaction spends or creates a slot a held intent's transaction
    /// does (two payouts drawing on the same pool slots, a coin spent twice):
    /// no block could carry both, and the template would be refused for them
    /// at every height without a culprit to forget.
    SlotConflict,
}

impl std::fmt::Display for HoldRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// The intents the pool holds: the highest attestation, one equivocation,
/// delegation objects by arrival.
#[derive(Clone, Debug, Default)]
pub struct GuardianPool {
    next_seq: u64,
    attestation: Option<HeldGuardian>,
    equivocation: Option<HeldGuardian>,
    delegation: Vec<HeldGuardian>,
    pub(crate) limiter: GuardianRateLimiter,
}

impl GuardianPool {
    /// In the template's order: the equivocation, the attestation, then the
    /// delegation objects by arrival.
    pub fn held(&self) -> impl Iterator<Item = &HeldGuardian> {
        self.equivocation
            .iter()
            .chain(self.attestation.iter())
            .chain(self.delegation.iter())
    }

    pub fn len(&self) -> usize {
        self.held().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether `intent` could be held beside what is (the conflicts of §10
    /// that need no rule).
    pub fn conflict(&self, intent: &GuardianIntent) -> Result<(), HoldRefusal> {
        if self.held().any(|held| held.intent.object == intent.object) {
            return Err(HoldRefusal::AlreadyHeld);
        }
        let slots = |pages: &[TxPage]| -> Vec<u32> {
            pages
                .iter()
                .flat_map(|page| {
                    page.body
                        .live_inputs()
                        .map(|(_, input)| input.slot_index)
                        .chain(
                            page.body
                                .live_outputs()
                                .map(|(_, output)| output.slot_index),
                        )
                        .collect::<Vec<u32>>()
                })
                .collect()
        };
        let own = slots(&intent.payment.pages);
        let replaced = |held: &HeldGuardian| {
            intent.object.kind == GuardianObjectKind::Attestation
                && held.intent.object.kind == GuardianObjectKind::Attestation
        };
        if self.held().filter(|held| !replaced(held)).any(|held| {
            slots(&held.intent.payment.pages)
                .iter()
                .any(|slot| own.contains(slot))
        }) {
            return Err(HoldRefusal::SlotConflict);
        }
        match intent.object.kind {
            GuardianObjectKind::Attestation => {
                let held = self
                    .attestation
                    .as_ref()
                    .and_then(|h| h.intent.attested_height());
                match (held, intent.attested_height()) {
                    (Some(held), Some(new)) if new <= held => Err(HoldRefusal::Covered { held }),
                    _ => Ok(()),
                }
            }
            GuardianObjectKind::Equivocation if self.equivocation.is_some() => {
                Err(HoldRefusal::EquivocationHeld)
            }
            GuardianObjectKind::Equivocation => Ok(()),
            _ if self.delegation.len() >= MAX_HELD_DELEGATION_OBJECTS => Err(HoldRefusal::Full),
            _ => Ok(()),
        }
    }

    /// Hold `intent` (a higher attestation replaces the one held); returns
    /// its arrival number.
    pub fn hold(
        &mut self,
        intent: GuardianIntent,
        group: GuardianGroupFacts,
    ) -> Result<u64, HoldRefusal> {
        self.conflict(&intent)?;
        let seq = self.next_seq;
        self.next_seq += 1;
        let kind = intent.object.kind;
        let held = HeldGuardian { seq, intent, group };
        match kind {
            GuardianObjectKind::Attestation => self.attestation = Some(held),
            GuardianObjectKind::Equivocation => self.equivocation = Some(held),
            _ => self.delegation.push(held),
        }
        Ok(seq)
    }

    /// Forget the intents numbered `seqs`; returns how many.
    pub fn forget(&mut self, seqs: &[u64]) -> usize {
        let before = self.len();
        self.attestation = self.attestation.take().filter(|h| !seqs.contains(&h.seq));
        self.equivocation = self.equivocation.take().filter(|h| !seqs.contains(&h.seq));
        self.delegation.retain(|h| !seqs.contains(&h.seq));
        before - self.len()
    }

    /// Forget everything (below J, or no guardian state to judge by).
    pub fn clear(&mut self) {
        self.attestation = None;
        self.equivocation = None;
        self.delegation.clear();
    }
}

/// What [`crate::AsyncMempool::submit_guardian`] admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuardianAdmitted {
    pub kind: GuardianObjectKind,
    pub marker: Address,
    /// Its arrival number in the pool.
    pub seq: u64,
    /// A reward funding before the latch whose pair (a funding by the other
    /// development fund) the pool does not hold yet: no block carries it
    /// before (F-4).
    pub pending_pair: bool,
}

#[cfg(test)]
mod tests;
