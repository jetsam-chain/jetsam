// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7 (T2.5): the destination objects this node holds for its miners.
//!
//! A destination object never travels alone, and never as a plain
//! transaction: it travels as an **intent** with the transaction that the
//! block carries for it.
//!
//! * A **committee update** (`CommitteeUpdate`, one per L1 period) with the
//!   transaction that pays for it and carries its `JTMLCU01` marker.
//! * A **release** (`VaultRelease` and its storage proof) with the finality
//!   update and the bridge account proof it was proven under, and its release
//!   group: the spend from the vault (published secret) that pays the burn's
//!   owner `amount − R` (D59), its `JTMVRL01` marker and the change.
//! * From the return-over-Base height on (D101, D107), a **release over
//!   Base**: the same, with the Base game proof (object 5) and the output root
//!   preimage (object 6) it was proven under, its account proof under the Base
//!   state root, and a release group that pays the creditor of the advance of
//!   exactly its burn, if the branch holds one (D107 V2, D114), else the
//!   owner. While that height is `None` (every profile today) its kind decodes
//!   as an unknown one, exactly as in a build without it.
//! * The fast lane of the return over Base (D127, D130, same height): a
//!   **fast release** (a release over Base with the relay's
//!   `FastGameAttestation`, its group carrying the fast lane's marker), a
//!   **contradiction** (`FastContradiction` with the slow Base objects it is
//!   read under, and the transaction carrying its `JTMFCT01` marker) and a
//!   **revocation** (`FastKeyRevocation` with the transaction carrying its
//!   `JTMFRV01` marker). Unknown kinds while the height is `None`.
//!
//! The pool checks at admission the shapes, the release group against the
//! vault gate, slots, then — A2-M1, with the committee keys this node holds
//! (`ChainView::committee_keys`) — the L1 light client and the storage proof
//! as the block does ([`verify_destination_intent`]), then the payment's
//! authorization. An intent it cannot verify is refused (softly when the
//! keys are missing: the node's own fault), so none is ever held, and a held
//! one is never replaced by one that does not verify. The template still
//! checks the whole destination section with `validate_block_destination`
//! before it mines, and builds again without the objects it refuses.

use std::collections::BTreeMap;

use jetsam_chain::consensus::destination::base_read::{
    decode_game_proof, decode_output_root_preimage,
};
use jetsam_chain::consensus::destination::forks::period_of_slot;
use jetsam_chain::consensus::destination::ssz::{LightClientFinalityUpdate, LightClientUpdate};
use jetsam_chain::consensus::destination::fast::{
    fast_marker_kind, fast_release_marker, revocation_marker, FastAttestation, FastContradiction,
    FastLaneState, FastMarkerKind, FastRevocation,
};
use jetsam_chain::consensus::destination::state::{
    committee_update_marker, verify_committee_update_payload, verify_fast_contradiction,
    verify_fast_release_intent, verify_fast_revocation, verify_releases_on_base,
    verify_releases_on_l1, BaseReleaseObjects, CommitteeKeyCache, CommitteeRef,
    DestinationBlockError, DestinationRules, ReopenedReleases,
};
use jetsam_chain::consensus::destination::{
    base_objects_on_the_wire, DestinationObject, DestinationObjectKind,
};
use jetsam_chain::consensus::vault::{
    decode_bridge_account_proof, destination_marker_kind, DestinationMarkerKind, ReleasedSet,
    VaultError, VaultRelease, VaultRules,
};
use jetsam_chain::consensus::wire_limits::{
    MAX_DESTINATION_SECTION_BYTES, MAX_TX_INTENT_BYTES_GLOBAL,
};
use jetsam_tx::{PagedSpendIntent, TxPage};

/// First bytes of an encoded destination intent.
pub const DESTINATION_INTENT_MAGIC: [u8; 8] = *b"JTMDIN01";
/// Largest encoded intent: its objects within one destination section, its
/// transaction within the global intent bound.
pub const MAX_DESTINATION_INTENT_BYTES: usize = DESTINATION_INTENT_MAGIC.len()
    + 1
    + 4 * 4
    + MAX_DESTINATION_SECTION_BYTES
    + MAX_TX_INTENT_BYTES_GLOBAL;
/// Committee updates held at once (one per period; a block carries one).
pub const MAX_HELD_COMMITTEE_UPDATES: usize = 4;
/// Releases held at once (one per withdrawal index `n`).
pub const MAX_HELD_RELEASES: usize = 64;
/// Contradictions held at once (one per withdrawal index `n`).
pub const MAX_HELD_CONTRADICTIONS: usize = 8;

const KIND_COMMITTEE_UPDATE: u8 = 1;
const KIND_RELEASE: u8 = 2;
const KIND_RELEASE_OVER_BASE: u8 = 3;
const KIND_FAST_RELEASE: u8 = 4;
const KIND_FAST_CONTRADICTION: u8 = 5;
const KIND_FAST_REVOCATION: u8 = 6;

/// A committee update with the transaction that pays for it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitteeUpdateIntent {
    /// The `LightClientUpdate` payload (SSZ).
    pub payload: Vec<u8>,
    /// Carries the update's `JTMLCU01` marker.
    pub payment: PagedSpendIntent,
}

impl CommitteeUpdateIntent {
    /// The period whose committee this update brings (attested period + 1).
    pub fn next_period(&self) -> Option<u64> {
        LightClientUpdate::decode_payload(&self.payload)
            .ok()
            .and_then(|update| period_of_slot(update.attested_header.beacon.slot).checked_add(1))
    }

    pub fn object(&self) -> DestinationObject {
        DestinationObject {
            kind: DestinationObjectKind::CommitteeUpdate,
            payload: self.payload.clone(),
        }
    }
}

/// A release with what proves it and the group that pays it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReleaseIntent {
    pub release: VaultRelease,
    /// The `LightClientFinalityUpdate` payload it is proven under.
    pub finality: Vec<u8>,
    /// The bridge account proof under that update's finalized root, then the
    /// storage proof of the contract's `genesisId` (A5-8): the
    /// `BridgeAccountProof` object payload.
    pub account_proof: Vec<u8>,
    /// The release group: a spend from the vault.
    pub payment: PagedSpendIntent,
    /// Over Base (from the return-over-Base height on): the game proof and
    /// the output root preimage; `account_proof` is then under the Base
    /// state root they open.
    pub base: Option<BaseReleaseProofs>,
    /// The fast lane: the `FastGameAttestation` payload (only over Base);
    /// the group then carries the fast lane's marker.
    pub fast: Option<Vec<u8>>,
}

/// A contradiction of a fast release, with the slow Base objects it is read
/// under and the transaction that carries its marker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastContradictionIntent {
    /// The `FastContradiction` payload.
    pub payload: Vec<u8>,
    pub finality: Vec<u8>,
    pub base: BaseReleaseProofs,
    /// The bridge account proof under the Base state root.
    pub account_proof: Vec<u8>,
    /// Carries the contradiction's `JTMFCT01` marker.
    pub payment: PagedSpendIntent,
}

impl FastContradictionIntent {
    /// The withdrawal index contradicted.
    pub fn n(&self) -> u64 {
        FastContradiction::decode_payload(&self.payload).map_or(0, |c| c.n)
    }

    /// The marker its transaction carries.
    pub fn marker(&self) -> Option<jetsam_poseidon2b::primitives::Address> {
        fast_markers_of(&self.payment)
            .into_iter()
            .find(|owner| fast_marker_kind(owner) == Some(FastMarkerKind::Contradiction))
    }

    pub fn object(&self) -> DestinationObject {
        DestinationObject {
            kind: DestinationObjectKind::FastContradiction,
            payload: self.payload.clone(),
        }
    }
}

/// A revocation of the relay's key by the cold key, with the transaction
/// that carries its marker.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastRevocationIntent {
    /// The `FastKeyRevocation` payload.
    pub payload: Vec<u8>,
    /// Carries the revocation's `JTMFRV01` marker.
    pub payment: PagedSpendIntent,
}

impl FastRevocationIntent {
    pub fn object(&self) -> DestinationObject {
        DestinationObject {
            kind: DestinationObjectKind::FastKeyRevocation,
            payload: self.payload.clone(),
        }
    }
}

/// The fast lane's markers a transaction's live outputs carry.
fn fast_markers_of(intent: &PagedSpendIntent) -> Vec<jetsam_poseidon2b::primitives::Address> {
    intent
        .pages
        .iter()
        .flat_map(|page| page.body.live_outputs().map(|(_, o)| o.owner))
        .filter(|owner| fast_marker_kind(owner).is_some())
        .collect()
}

/// What a release over Base is proven under beside its finality update.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseReleaseProofs {
    /// The `BaseGameProof` object payload (deduplicated nodes, D107 §9.2).
    pub game_proof: Vec<u8>,
    /// The `OutputRootPreimage` object payload.
    pub output_preimage: Vec<u8>,
}

impl ReleaseIntent {
    /// What its group pays first: the payee V2 reads.
    pub fn paid_to(&self) -> Option<jetsam_poseidon2b::primitives::Address> {
        self.payment
            .pages
            .iter()
            .flat_map(|page| page.body.live_outputs().map(|(_, output)| output.owner))
            .next()
    }

    /// Whether it shares the block's proofs with `other`: one finality
    /// update, one account proof and, over Base, one game proof and preimage
    /// per block, and one attestation (or none) — one mode per block.
    pub fn shares_proofs_with(&self, other: &Self) -> bool {
        self.finality == other.finality
            && self.account_proof == other.account_proof
            && self.base == other.base
            && self.fast == other.fast
    }

    /// The marker its group carries: the release's, or for a fast release
    /// the fast lane's (the attested L2 block).
    pub fn marker(&self) -> Option<jetsam_poseidon2b::primitives::Address> {
        match &self.fast {
            None => Some(self.release.marker()),
            Some(payload) => FastAttestation::decode_payload(payload).ok().map(|a| {
                fast_release_marker(
                    self.release.n,
                    &self.release.owner,
                    self.release.amount,
                    a.statement.l2_block,
                )
            }),
        }
    }

    /// The `FastGameAttestation` object, for a fast release.
    pub fn attestation_object(&self) -> Option<DestinationObject> {
        self.fast.as_ref().map(|payload| DestinationObject {
            kind: DestinationObjectKind::FastGameAttestation,
            payload: payload.clone(),
        })
    }

    /// D52, D130: what the next block refuses of this release for its paid
    /// set, on a branch that paid `released` and whose fast lane is `lane` —
    /// the block's own rule ([`ReopenedReleases`]): its `n` paid, unless a
    /// contradiction reopened it and this is a slow release of another burn
    /// than the fast lane paid (once: the record is consumed).
    pub fn check_unpaid(
        &self,
        released: &ReleasedSet,
        lane: &FastLaneState,
    ) -> Result<(), DestinationBlockError> {
        let r = &self.release;
        let reopened = ReopenedReleases::of_block(lane, self.fast.is_some(), &[]);
        if reopened.paid_set(released).contains(r.n) {
            return Err(DestinationBlockError::Vault(VaultError::ReleaseAlreadyPaid {
                n: r.n,
            }));
        }
        reopened
            .consumed_by(&[(r.n, r.owner, r.amount)])
            .map(|_| ())
    }
}

/// One destination object with its transaction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DestinationIntent {
    CommitteeUpdate(CommitteeUpdateIntent),
    Release(ReleaseIntent),
    FastContradiction(FastContradictionIntent),
    FastRevocation(FastRevocationIntent),
}

/// Why bytes are not a destination intent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DestinationIntentError {
    TooLarge {
        actual: usize,
        max: usize,
    },
    BadMagic,
    UnknownKind(u8),
    Truncated,
    TrailingBytes,
    /// An object payload over the section budget, or empty.
    ObjectSize,
    /// A payload that does not decode as its object.
    MalformedObject(DestinationObjectKind),
    /// The transaction does not decode, or is not this intent's.
    Payment(String),
}

impl std::fmt::Display for DestinationIntentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

fn put(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn take<'a>(src: &mut &'a [u8]) -> Result<&'a [u8], DestinationIntentError> {
    if src.len() < 4 {
        return Err(DestinationIntentError::Truncated);
    }
    let len = u32::from_le_bytes(src[..4].try_into().expect("four bytes")) as usize;
    let rest = &src[4..];
    if rest.len() < len {
        return Err(DestinationIntentError::Truncated);
    }
    let (head, tail) = rest.split_at(len);
    *src = tail;
    Ok(head)
}

fn object_bytes<'a>(src: &mut &'a [u8]) -> Result<&'a [u8], DestinationIntentError> {
    let bytes = take(src)?;
    if bytes.is_empty() || bytes.len() > MAX_DESTINATION_SECTION_BYTES {
        return Err(DestinationIntentError::ObjectSize);
    }
    Ok(bytes)
}

fn payment_of(bytes: &[u8]) -> Result<PagedSpendIntent, DestinationIntentError> {
    if bytes.len() > MAX_TX_INTENT_BYTES_GLOBAL {
        return Err(DestinationIntentError::TooLarge {
            actual: bytes.len(),
            max: MAX_TX_INTENT_BYTES_GLOBAL,
        });
    }
    let mut src = bytes;
    let intent = PagedSpendIntent::decode(&mut src)
        .map_err(|e| DestinationIntentError::Payment(format!("{e:?}")))?;
    if !src.is_empty() {
        return Err(DestinationIntentError::TrailingBytes);
    }
    Ok(intent)
}

impl DestinationIntent {
    /// `magic ‖ kind ‖ (len:u32 LE ‖ bytes)*`: a committee update carries its
    /// payload and its payment; a release its object payload, finality
    /// update, account proof and release group; a release over Base its
    /// object payload, finality update, game proof, output root preimage,
    /// account proof and release group.
    pub fn encode(&self) -> Result<Vec<u8>, DestinationIntentError> {
        let mut out = DESTINATION_INTENT_MAGIC.to_vec();
        let payment = |intent: &PagedSpendIntent| {
            intent
                .to_bytes()
                .map_err(|e| DestinationIntentError::Payment(format!("{e:?}")))
        };
        match self {
            Self::CommitteeUpdate(update) => {
                out.push(KIND_COMMITTEE_UPDATE);
                put(&mut out, &update.payload);
                put(&mut out, &payment(&update.payment)?);
            }
            Self::Release(release) => {
                out.push(match (&release.base, &release.fast) {
                    (None, _) => KIND_RELEASE,
                    (Some(_), None) => KIND_RELEASE_OVER_BASE,
                    (Some(_), Some(_)) => KIND_FAST_RELEASE,
                });
                put(&mut out, &release.release.encode_payload());
                put(&mut out, &release.finality);
                if let Some(base) = &release.base {
                    put(&mut out, &base.game_proof);
                    put(&mut out, &base.output_preimage);
                    if let Some(attestation) = &release.fast {
                        put(&mut out, attestation);
                    }
                }
                put(&mut out, &release.account_proof);
                put(&mut out, &payment(&release.payment)?);
            }
            Self::FastContradiction(c) => {
                out.push(KIND_FAST_CONTRADICTION);
                put(&mut out, &c.payload);
                put(&mut out, &c.finality);
                put(&mut out, &c.base.game_proof);
                put(&mut out, &c.base.output_preimage);
                put(&mut out, &c.account_proof);
                put(&mut out, &payment(&c.payment)?);
            }
            Self::FastRevocation(r) => {
                out.push(KIND_FAST_REVOCATION);
                put(&mut out, &r.payload);
                put(&mut out, &payment(&r.payment)?);
            }
        }
        Ok(out)
    }

    /// The canonical decoding, bounded before anything is allocated beyond
    /// the input; each object payload is decoded as its object. A release
    /// over Base decodes only once this build's return-over-Base height is
    /// armed ([`base_objects_on_the_wire`]): before, its kind is unknown.
    pub fn decode(bytes: &[u8]) -> Result<Self, DestinationIntentError> {
        Self::decode_with(bytes, base_objects_on_the_wire())
    }

    /// [`Self::decode`], a release over Base read only if `over_base` (the
    /// destination rules arm a return-over-Base height).
    pub fn decode_with(bytes: &[u8], over_base: bool) -> Result<Self, DestinationIntentError> {
        if bytes.len() > MAX_DESTINATION_INTENT_BYTES {
            return Err(DestinationIntentError::TooLarge {
                actual: bytes.len(),
                max: MAX_DESTINATION_INTENT_BYTES,
            });
        }
        let Some(rest) = bytes.strip_prefix(&DESTINATION_INTENT_MAGIC) else {
            return Err(DestinationIntentError::BadMagic);
        };
        let (&kind, mut src) = rest
            .split_first()
            .ok_or(DestinationIntentError::Truncated)?;
        let intent = match kind {
            KIND_COMMITTEE_UPDATE => {
                let payload = object_bytes(&mut src)?;
                LightClientUpdate::decode_payload(payload).map_err(|_| {
                    DestinationIntentError::MalformedObject(DestinationObjectKind::CommitteeUpdate)
                })?;
                let payment = payment_of(take(&mut src)?)?;
                Self::CommitteeUpdate(CommitteeUpdateIntent {
                    payload: payload.to_vec(),
                    payment,
                })
            }
            KIND_RELEASE | KIND_RELEASE_OVER_BASE | KIND_FAST_RELEASE
                if kind == KIND_RELEASE || over_base =>
            {
                let release =
                    VaultRelease::decode_payload(object_bytes(&mut src)?).map_err(|_| {
                        DestinationIntentError::MalformedObject(DestinationObjectKind::VaultRelease)
                    })?;
                let finality = object_bytes(&mut src)?;
                LightClientFinalityUpdate::decode_payload(finality).map_err(|_| {
                    DestinationIntentError::MalformedObject(DestinationObjectKind::FinalityUpdate)
                })?;
                let base = if kind != KIND_RELEASE {
                    Some(base_proofs(&mut src)?)
                } else {
                    None
                };
                let fast = if kind == KIND_FAST_RELEASE {
                    let attestation = object_bytes(&mut src)?;
                    FastAttestation::decode_payload(attestation).map_err(|_| {
                        DestinationIntentError::MalformedObject(
                            DestinationObjectKind::FastGameAttestation,
                        )
                    })?;
                    Some(attestation.to_vec())
                } else {
                    None
                };
                // The account proof, then the genesis proof (A5-8).
                let account_proof = object_bytes(&mut src)?;
                decode_bridge_account_proof(account_proof).map_err(|_| {
                    DestinationIntentError::MalformedObject(
                        DestinationObjectKind::BridgeAccountProof,
                    )
                })?;
                let payment = payment_of(take(&mut src)?)?;
                Self::Release(ReleaseIntent {
                    release,
                    finality: finality.to_vec(),
                    account_proof: account_proof.to_vec(),
                    payment,
                    base,
                    fast,
                })
            }
            KIND_FAST_CONTRADICTION if over_base => {
                let payload = object_bytes(&mut src)?;
                FastContradiction::decode_payload(payload).map_err(|_| {
                    DestinationIntentError::MalformedObject(
                        DestinationObjectKind::FastContradiction,
                    )
                })?;
                let finality = object_bytes(&mut src)?;
                LightClientFinalityUpdate::decode_payload(finality).map_err(|_| {
                    DestinationIntentError::MalformedObject(DestinationObjectKind::FinalityUpdate)
                })?;
                let base = base_proofs(&mut src)?;
                let account_proof = object_bytes(&mut src)?;
                decode_bridge_account_proof(account_proof).map_err(|_| {
                    DestinationIntentError::MalformedObject(
                        DestinationObjectKind::BridgeAccountProof,
                    )
                })?;
                let payment = payment_of(take(&mut src)?)?;
                Self::FastContradiction(FastContradictionIntent {
                    payload: payload.to_vec(),
                    finality: finality.to_vec(),
                    base,
                    account_proof: account_proof.to_vec(),
                    payment,
                })
            }
            KIND_FAST_REVOCATION if over_base => {
                let payload = object_bytes(&mut src)?;
                FastRevocation::decode_payload(payload).map_err(|_| {
                    DestinationIntentError::MalformedObject(
                        DestinationObjectKind::FastKeyRevocation,
                    )
                })?;
                let payment = payment_of(take(&mut src)?)?;
                Self::FastRevocation(FastRevocationIntent {
                    payload: payload.to_vec(),
                    payment,
                })
            }
            other => return Err(DestinationIntentError::UnknownKind(other)),
        };
        if !src.is_empty() {
            return Err(DestinationIntentError::TrailingBytes);
        }
        intent.check_payment_markers()?;
        Ok(intent)
    }

    /// The transaction is this object's: a committee update's payment carries
    /// exactly one v1.7 marker, its own `JTMLCU01`; a release's group carries
    /// its own `JTMVRL01` (a fast release, the fast lane's commitment), and no
    /// fast-lane marker (the rest of its shape is the pool's admission); a
    /// contradiction's or a revocation's transaction carries no v1.7 marker
    /// and exactly one fast-lane marker, of its kind (a revocation's opens
    /// its key; a contradiction's is judged against the branch's record).
    fn check_payment_markers(&self) -> Result<(), DestinationIntentError> {
        let markers = |intent: &PagedSpendIntent| -> Vec<jetsam_poseidon2b::primitives::Address> {
            intent
                .pages
                .iter()
                .flat_map(|page| page.body.live_outputs().map(|(_, o)| o.owner))
                .filter(|owner| destination_marker_kind(owner).is_some())
                .collect()
        };
        let ok = match self {
            Self::CommitteeUpdate(update) => {
                markers(&update.payment) == [committee_update_marker(&update.payload)]
                    && fast_markers_of(&update.payment).is_empty()
            }
            Self::Release(release) => {
                release.marker().is_some_and(|m| markers(&release.payment) == [m])
                    && fast_markers_of(&release.payment).is_empty()
            }
            Self::FastContradiction(c) => {
                markers(&c.payment).is_empty()
                    && fast_markers_of(&c.payment).len() == 1
                    && c.marker().is_some()
            }
            Self::FastRevocation(r) => {
                markers(&r.payment).is_empty()
                    && FastRevocation::decode_payload(&r.payload).is_ok_and(|rev| {
                        fast_markers_of(&r.payment) == [revocation_marker(&rev.key)]
                    })
            }
        };
        if ok {
            Ok(())
        } else {
            Err(DestinationIntentError::Payment(
                "the transaction does not carry exactly this object's marker".into(),
            ))
        }
    }

    pub fn payment(&self) -> &PagedSpendIntent {
        match self {
            Self::CommitteeUpdate(update) => &update.payment,
            Self::Release(release) => &release.payment,
            Self::FastContradiction(c) => &c.payment,
            Self::FastRevocation(r) => &r.payment,
        }
    }
}

/// A game proof and an output root preimage, each decoded as its object.
fn base_proofs(src: &mut &[u8]) -> Result<BaseReleaseProofs, DestinationIntentError> {
    let game_proof = object_bytes(src)?;
    decode_game_proof(game_proof).map_err(|_| {
        DestinationIntentError::MalformedObject(DestinationObjectKind::BaseGameProof)
    })?;
    let output_preimage = object_bytes(src)?;
    decode_output_root_preimage(output_preimage).map_err(|_| {
        DestinationIntentError::MalformedObject(DestinationObjectKind::OutputRootPreimage)
    })?;
    Ok(BaseReleaseProofs {
        game_proof: game_proof.to_vec(),
        output_preimage: output_preimage.to_vec(),
    })
}

/// What [`crate::AsyncMempool::submit_destination`] admitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DestinationAdmitted {
    /// A committee update bringing the committee of `period`.
    CommitteeUpdate { period: u64 },
    /// The release of withdrawal index `n`.
    Release { n: u64 },
    /// The release of withdrawal index `n` over Base.
    ReleaseOverBase { n: u64 },
    /// The fast release of withdrawal index `n` (the fast lane).
    FastRelease { n: u64 },
    /// A contradiction of the fast release of `n`.
    FastContradiction { n: u64 },
    /// A revocation of the relay's key.
    FastRevocation,
}

/// What the pool holds: committee updates by the period they bring, releases
/// by withdrawal index.
#[derive(Clone, Debug, Default)]
pub struct DestinationPool {
    committee_updates: BTreeMap<u64, CommitteeUpdateIntent>,
    releases: BTreeMap<u64, ReleaseIntent>,
    contradictions: BTreeMap<u64, FastContradictionIntent>,
    revocation: Option<FastRevocationIntent>,
}

/// Why the pool does not hold an intent it was offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldError {
    /// No room left for this kind.
    Full,
    /// A committee update whose period does not decode.
    NoPeriod,
}

/// A2-M1: the L1 rules of the block for one destination intent, on a branch
/// that knows `committees` with the keys `keys` this node holds — what
/// `validate_block_destination` asks of the object beside the vault gate: a
/// committee update is the anchor update or is signed by the last committee
/// known ([`verify_committee_update_payload`]); a release's finality update
/// is signed by a known committee, the bridge account opens its finalized
/// root and the storage proof proves the burn (V7,
/// [`verify_releases_on_l1`]); a release over Base, its game, output root,
/// bridge account and storage proof under the Base state root
/// ([`verify_releases_on_base`]). `rules` are those of the next block
/// (`DestinationRules::effective_at`: from the return-over-Base height on,
/// the light client reads the L1 Base settles on). Pure; BLS for every kind.
pub fn verify_destination_intent(
    intent: &DestinationIntent,
    committees: &[CommitteeRef],
    rules: &DestinationRules,
    lane: &FastLaneState,
    keys: &CommitteeKeyCache,
) -> Result<(), DestinationBlockError> {
    match intent {
        DestinationIntent::Release(ReleaseIntent {
            release,
            finality,
            account_proof,
            base: Some(base),
            fast: Some(attestation),
            ..
        }) => verify_fast_release_intent(
            &BaseReleaseObjects {
                finality,
                game_proof: &base.game_proof,
                output_preimage: &base.output_preimage,
                account_proof,
            },
            attestation,
            std::slice::from_ref(release),
            committees,
            rules,
            lane,
            &CommitteeKeyCache::default(),
            keys,
        )
        .map(|_| ()),
        DestinationIntent::FastContradiction(c) => verify_fast_contradiction(
            &BaseReleaseObjects {
                finality: &c.finality,
                game_proof: &c.base.game_proof,
                output_preimage: &c.base.output_preimage,
                account_proof: &c.account_proof,
            },
            &c.payload,
            c.marker().ok_or(DestinationBlockError::FastObjectsDoNotMatchMarkers)?,
            committees,
            rules,
            lane,
            &CommitteeKeyCache::default(),
            keys,
        )
        .map(|_| ()),
        DestinationIntent::FastRevocation(r) => {
            verify_fast_revocation(&r.payload, rules, lane).map(|_| ())
        }
        DestinationIntent::CommitteeUpdate(update) => {
            verify_committee_update_payload(&update.payload, committees, rules, keys).map(|_| ())
        }
        DestinationIntent::Release(ReleaseIntent {
            release,
            finality,
            account_proof,
            base: Some(base),
            ..
        }) => verify_releases_on_base(
            &BaseReleaseObjects {
                finality,
                game_proof: &base.game_proof,
                output_preimage: &base.output_preimage,
                account_proof,
            },
            std::slice::from_ref(release),
            committees,
            rules,
            &CommitteeKeyCache::default(),
            keys,
        )
        .map(|_| ()),
        DestinationIntent::Release(release) => verify_releases_on_l1(
            &release.finality,
            &release.account_proof,
            std::slice::from_ref(&release.release),
            committees,
            rules,
            &CommitteeKeyCache::default(),
            keys,
        ),
    }
}

/// A2-M2: the vault rules of a transaction that pays for an object other
/// than a release (a committee update's, a v1.5 client object's, a guardian
/// object's), as the vault gate judges its group: what
/// [`check_plain_vault_transaction`] asks of a plain transaction — no spend
/// from the vault (V1), no destination marker carrying value, no release
/// marker, no output to the vault and no deposit marker (V4: an object
/// payment is never a deposit, its own marker beside a deposit marker being
/// two markers in one group, M-3) — a committee-update marker aside (the
/// update's own, which the intent's decoding pins). The caller applies it
/// from J on.
///
/// [`check_plain_vault_transaction`]: jetsam_chain::consensus::vault::check_plain_vault_transaction
pub fn check_object_payment_vault(pages: &[TxPage], rules: &VaultRules) -> Result<(), VaultError> {
    if pages
        .iter()
        .any(|page| page.body.input_owner == rules.vault)
    {
        return Err(VaultError::SpendFromLockedAddress { group: 0 });
    }
    for page in pages {
        for (_, output) in page.body.live_outputs() {
            match destination_marker_kind(&output.owner) {
                Some(_) if output.amount != 0 => {
                    return Err(VaultError::MarkerCarriesValue { group: 0 })
                }
                Some(DestinationMarkerKind::Release) => {
                    return Err(VaultError::ReleaseMarkerOutsideRelease { group: 0 })
                }
                Some(DestinationMarkerKind::Deposit) => {
                    return Err(VaultError::DepositShape { group: 0 })
                }
                Some(DestinationMarkerKind::LightClientUpdate) | None => {}
            }
            if output.owner == rules.vault {
                return Err(VaultError::VaultOutputOutsideDeposit { group: 0 });
            }
        }
    }
    Ok(())
}

impl DestinationPool {
    /// Hold `intent` — verified at admission ([`verify_destination_intent`]):
    /// no intent that does not verify is ever held, so none replaces a held
    /// one — replacing the one of the same period or index (the newer proof
    /// wins: a relayer refreshes a release under a later finality update). A
    /// full release pool makes room for a lower index than its largest.
    pub fn hold(&mut self, intent: DestinationIntent) -> Result<(), HoldError> {
        match intent {
            DestinationIntent::CommitteeUpdate(update) => {
                let period = update.next_period().ok_or(HoldError::NoPeriod)?;
                if !self.committee_updates.contains_key(&period)
                    && self.committee_updates.len() >= MAX_HELD_COMMITTEE_UPDATES
                {
                    return Err(HoldError::Full);
                }
                self.committee_updates.insert(period, update);
            }
            DestinationIntent::Release(release) => {
                let n = release.release.n;
                if !self.releases.contains_key(&n) && self.releases.len() >= MAX_HELD_RELEASES {
                    // Full: the largest index gives way to a lower one (the
                    // template serves releases by increasing `n`, A2-M1).
                    match self.releases.last_key_value() {
                        Some((&largest, _)) if n < largest => {
                            self.releases.remove(&largest);
                        }
                        _ => return Err(HoldError::Full),
                    }
                }
                self.releases.insert(n, release);
            }
            DestinationIntent::FastContradiction(c) => {
                let n = c.n();
                if !self.contradictions.contains_key(&n)
                    && self.contradictions.len() >= MAX_HELD_CONTRADICTIONS
                {
                    return Err(HoldError::Full);
                }
                self.contradictions.insert(n, c);
            }
            DestinationIntent::FastRevocation(r) => {
                self.revocation = Some(r);
            }
        }
        Ok(())
    }

    /// Contradictions by increasing `n`.
    pub fn contradictions(&self) -> impl Iterator<Item = &FastContradictionIntent> {
        self.contradictions.values()
    }

    /// The revocation held, if any.
    pub fn revocation(&self) -> Option<&FastRevocationIntent> {
        self.revocation.as_ref()
    }

    /// Forget the fast lane's intents `lane` makes useless: a revocation of a
    /// closed key, a contradiction of an `n` without an open record.
    pub fn prune_fast(&mut self, lane: &FastLaneState) -> usize {
        let before = self.len();
        if self.revocation.as_ref().is_some_and(|r| {
            FastRevocation::decode_payload(&r.payload).map_or(true, |rev| lane.is_closed(&rev.key))
        }) {
            self.revocation = None;
        }
        self.contradictions
            .retain(|n, _| lane.records.get(n).is_some_and(|record| !record.contradicted));
        before - self.len()
    }

    /// Releases by increasing `n` (the template's policy, porte R §3.3).
    pub fn releases(&self) -> impl Iterator<Item = &ReleaseIntent> {
        self.releases.values()
    }

    /// Committee updates by increasing period.
    pub fn committee_updates(&self) -> impl Iterator<Item = &CommitteeUpdateIntent> {
        self.committee_updates.values()
    }

    pub fn len(&self) -> usize {
        self.committee_updates.len()
            + self.releases.len()
            + self.contradictions.len()
            + usize::from(self.revocation.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Forget what a branch no longer needs: releases `paid` says the branch
    /// paid or that `stale` says can no longer be built (inputs spent),
    /// committee updates for a period at or below `known_period`. Returns
    /// the number forgotten.
    pub fn prune(
        &mut self,
        paid: impl Fn(&ReleaseIntent) -> bool,
        stale: impl Fn(&ReleaseIntent) -> bool,
        known_period: Option<u64>,
    ) -> usize {
        let before = self.len();
        self.releases
            .retain(|_, release| !paid(release) && !stale(release));
        if let Some(known) = known_period {
            self.committee_updates.retain(|period, _| *period > known);
        }
        before - self.len()
    }

    /// Forget the releases of `releases` and the committee update bringing
    /// `committee_period` (a template's block was refused for them).
    pub fn forget(&mut self, releases: &[u64], committee_period: Option<u64>) -> usize {
        let before = self.len();
        self.releases.retain(|n, _| !releases.contains(n));
        if let Some(period) = committee_period {
            self.committee_updates.remove(&period);
        }
        before - self.len()
    }

    /// Forget the contradictions of `contradictions` and, if `revocation`,
    /// the revocation (a template's block was refused for them).
    pub fn forget_fast(&mut self, contradictions: &[u64], revocation: bool) -> usize {
        let before = self.len();
        self.contradictions.retain(|n, _| !contradictions.contains(n));
        if revocation {
            self.revocation = None;
        }
        before - self.len()
    }

    /// Forget everything (below J, or a view the pool cannot judge by).
    pub fn clear(&mut self) {
        self.committee_updates.clear();
        self.releases.clear();
        self.contradictions.clear();
        self.revocation = None;
    }
}

#[cfg(test)]
pub(crate) mod tests;
