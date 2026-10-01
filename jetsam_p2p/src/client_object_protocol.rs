// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.5 client objects between nodes (M3 task 3.7): client proofs and the
//! client matrices registered on chain, moved by request-response.
//!
//! A client proof (~409 kB measured, 509 009 B bound) does not fit the
//! gossip cap (303 495 B, one PagedSpend intent), and a registered matrix
//! file (up to 16 MiB) fits no single response. So:
//!
//! - a node that holds a client proof **announces** it in a few dozen bytes
//!   ([`ClientProofAnnouncement`]: the submission, the bundle's byte digest
//!   and length, the fee it pays) — gossip-sized;
//! - a node that wants it **requests** the [`ClientProofBundle`] (submission,
//!   paying transaction, proof) by that id from a peer that announced it; the
//!   response is refused unless its length and byte digest are exactly the
//!   id's, before anything else reads it; the bundle is then decoded under its
//!   bounds (a proof longer than the client form's bound is refused before it
//!   is read), its payment must name its submission, and the node pre-passes
//!   the proof once (`PreparedHistoryStepClient`) and queues it
//!   (`ClientSubmissionQueue`);
//! - a registered matrix file is fetched as a **manifest** (its chunk
//!   digests, checked against the registered file root) then **chunks** of
//!   1 MiB, each checked against its digest, from any mix of peers.
//!
//! [`ClientProofFetcher`] is the requester's policy: providers learnt from
//! announcements, one request per proof at a time, a per-peer cap, and a
//! peer that serves bytes failing their id is excluded for that proof and
//! reported for penalty — never a peer that merely timed out or did not
//! have it, and never the other providers of the same proof.

use std::collections::{BTreeMap, BTreeSet};

use jetsam_chain::consensus::client_objects::{
    matrix_file_chunk_count, matrix_file_chunk_digest, matrix_file_root_from_chunk_digests,
    pays_submission, ClientRegistryEntry, ClientSubmission, CLIENT_MATRIX_CHUNK_BYTES,
    CLIENT_MATRIX_MAX_FILE_BYTES, CLIENT_PROOF_MAX_WIRE_BYTES,
};
use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_tx::{PagedSpendIntent, MAX_PAGED_SPEND_INTENT_BYTES};

pub type Hash32 = [u8; 32];

const BUNDLE_MAGIC: [u8; 4] = *b"JCP1";
const ANNOUNCEMENT_MAGIC: [u8; 4] = *b"JCA1";
const BUNDLE_DIGEST_DOMAIN: &[u8] = b"JTM/P2P/CLIENT-PROOF-BUNDLE/V1";

/// Fixed bytes of a bundle: magic, submission, the two length prefixes.
pub const CLIENT_PROOF_BUNDLE_FIXED_BYTES: usize = 4 + 64 + 4 + 4;

/// Largest bundle: one maximal payment intent and one maximal proof.
pub const MAX_CLIENT_PROOF_BUNDLE_BYTES: usize =
    CLIENT_PROOF_BUNDLE_FIXED_BYTES + MAX_PAGED_SPEND_INTENT_BYTES + CLIENT_PROOF_MAX_WIRE_BYTES;

/// Exact size of an announcement.
pub const CLIENT_PROOF_ANNOUNCEMENT_BYTES: usize = 4 + 64 + 32 + 4 + 8;

/// Largest matrix manifest: one digest per 1 MiB chunk of the largest file.
pub const MAX_CLIENT_MATRIX_MANIFEST_BYTES: usize =
    32 * matrix_file_chunk_count(CLIENT_MATRIX_MAX_FILE_BYTES);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientTransportError {
    Truncated,
    TrailingBytes,
    BadMagic,
    /// A declared proof longer than the client form's bound (π too large).
    ProofTooLarge {
        actual: usize,
        max: usize,
    },
    EmptyProof,
    /// A declared payment intent longer than one intent may be.
    PaymentTooLarge {
        actual: usize,
        max: usize,
    },
    /// The payment is not a canonical intent.
    MalformedPayment,
    /// The payment does not name this submission.
    NotThePayment,
    /// The bytes are not those of the requested id.
    WrongBytes,
}

impl std::fmt::Display for ClientTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ClientTransportError {}

// ---------------------------------------------------------------------------
// Client proof bundle and its id
// ---------------------------------------------------------------------------

/// Network identity of one client proof bundle: its submission and the
/// digest and length of its exact bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClientProofId {
    pub submission: ClientSubmission,
    pub bundle_digest: Hash32,
    pub encoded_len: u32,
}

impl ClientProofId {
    pub fn of_bytes(submission: ClientSubmission, bytes: &[u8]) -> Option<Self> {
        Some(Self {
            submission,
            bundle_digest: poseidon2b_hash_bytes(BUNDLE_DIGEST_DOMAIN, bytes),
            encoded_len: u32::try_from(bytes.len()).ok()?,
        })
    }

    pub fn matches_bytes(&self, bytes: &[u8]) -> bool {
        usize::try_from(self.encoded_len).ok() == Some(bytes.len())
            && self.bundle_digest == poseidon2b_hash_bytes(BUNDLE_DIGEST_DOMAIN, bytes)
    }
}

/// What travels for one submission: the submission, the ordinary transaction
/// that pays it (a complete intent with its authorization), and the client
/// proof (`encode_history_step_client_proof` bytes, opaque here).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientProofBundle {
    pub submission: ClientSubmission,
    pub payment: PagedSpendIntent,
    pub proof_bytes: Vec<u8>,
}

impl ClientProofBundle {
    pub fn new(
        submission: ClientSubmission,
        payment: PagedSpendIntent,
        proof_bytes: Vec<u8>,
    ) -> Result<Self, ClientTransportError> {
        check_proof_len(proof_bytes.len())?;
        let payment_len = payment
            .to_bytes()
            .map_err(|_| ClientTransportError::MalformedPayment)?
            .len();
        if payment_len > MAX_PAGED_SPEND_INTENT_BYTES {
            return Err(ClientTransportError::PaymentTooLarge {
                actual: payment_len,
                max: MAX_PAGED_SPEND_INTENT_BYTES,
            });
        }
        if !pays_submission(&payment.pages, &submission) {
            return Err(ClientTransportError::NotThePayment);
        }
        Ok(Self {
            submission,
            payment,
            proof_bytes,
        })
    }

    pub fn encode(&self) -> Vec<u8> {
        let payment = self
            .payment
            .to_bytes()
            .expect("a bundle's payment is a canonical intent");
        let mut bytes = Vec::with_capacity(
            CLIENT_PROOF_BUNDLE_FIXED_BYTES + payment.len() + self.proof_bytes.len(),
        );
        bytes.extend_from_slice(&BUNDLE_MAGIC);
        bytes.extend_from_slice(&self.submission.matrix_digest);
        bytes.extend_from_slice(&self.submission.io_commitment);
        bytes.extend_from_slice(&(payment.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payment);
        bytes.extend_from_slice(&(self.proof_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.proof_bytes);
        bytes
    }

    /// Decode a bundle, every length bounded before it is read.
    pub fn decode(bytes: &[u8]) -> Result<Self, ClientTransportError> {
        let mut src = bytes;
        if take(&mut src, 4)? != BUNDLE_MAGIC {
            return Err(ClientTransportError::BadMagic);
        }
        let submission = ClientSubmission {
            matrix_digest: take_hash(&mut src)?,
            io_commitment: take_hash(&mut src)?,
        };
        let payment_len = take_u32(&mut src)? as usize;
        if payment_len > MAX_PAGED_SPEND_INTENT_BYTES {
            return Err(ClientTransportError::PaymentTooLarge {
                actual: payment_len,
                max: MAX_PAGED_SPEND_INTENT_BYTES,
            });
        }
        let payment = PagedSpendIntent::from_bytes(take(&mut src, payment_len)?)
            .map_err(|_| ClientTransportError::MalformedPayment)?;
        let proof_len = take_u32(&mut src)? as usize;
        check_proof_len(proof_len)?;
        let proof_bytes = take(&mut src, proof_len)?.to_vec();
        if !src.is_empty() {
            return Err(ClientTransportError::TrailingBytes);
        }
        if !pays_submission(&payment.pages, &submission) {
            return Err(ClientTransportError::NotThePayment);
        }
        Ok(Self {
            submission,
            payment,
            proof_bytes,
        })
    }

    /// Decode the bytes received for `id`: exactly its bytes, then exactly its
    /// submission.
    pub fn decode_for(id: &ClientProofId, bytes: &[u8]) -> Result<Self, ClientTransportError> {
        if !id.matches_bytes(bytes) {
            return Err(ClientTransportError::WrongBytes);
        }
        let bundle = Self::decode(bytes)?;
        if bundle.submission != id.submission {
            return Err(ClientTransportError::NotThePayment);
        }
        Ok(bundle)
    }

    pub fn id(&self) -> ClientProofId {
        let bytes = self.encode();
        ClientProofId::of_bytes(self.submission, &bytes)
            .expect("a bundle is bounded below u32::MAX bytes")
    }
}

/// Gossip-sized notice that a node holds one client proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClientProofAnnouncement {
    pub id: ClientProofId,
    /// The paying transaction's fee, the miner's ranking key.
    pub fee: u64,
}

impl ClientProofAnnouncement {
    pub fn encode(&self) -> [u8; CLIENT_PROOF_ANNOUNCEMENT_BYTES] {
        let mut bytes = [0u8; CLIENT_PROOF_ANNOUNCEMENT_BYTES];
        bytes[..4].copy_from_slice(&ANNOUNCEMENT_MAGIC);
        bytes[4..36].copy_from_slice(&self.id.submission.matrix_digest);
        bytes[36..68].copy_from_slice(&self.id.submission.io_commitment);
        bytes[68..100].copy_from_slice(&self.id.bundle_digest);
        bytes[100..104].copy_from_slice(&self.id.encoded_len.to_le_bytes());
        bytes[104..112].copy_from_slice(&self.fee.to_le_bytes());
        bytes
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ClientTransportError> {
        match bytes.len().cmp(&CLIENT_PROOF_ANNOUNCEMENT_BYTES) {
            std::cmp::Ordering::Less => return Err(ClientTransportError::Truncated),
            std::cmp::Ordering::Greater => return Err(ClientTransportError::TrailingBytes),
            std::cmp::Ordering::Equal => {}
        }
        if bytes[..4] != ANNOUNCEMENT_MAGIC {
            return Err(ClientTransportError::BadMagic);
        }
        let hash = |offset: usize| -> Hash32 { bytes[offset..offset + 32].try_into().unwrap() };
        Ok(Self {
            id: ClientProofId {
                submission: ClientSubmission {
                    matrix_digest: hash(4),
                    io_commitment: hash(36),
                },
                bundle_digest: hash(68),
                encoded_len: u32::from_le_bytes(bytes[100..104].try_into().unwrap()),
            },
            fee: u64::from_le_bytes(bytes[104..112].try_into().unwrap()),
        })
    }
}

// ---------------------------------------------------------------------------
// Registered matrix files
// ---------------------------------------------------------------------------

/// Network identity of a registered matrix file (from its registry entry).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MatrixFileId {
    pub matrix_digest: Hash32,
    pub file_root: Hash32,
    pub file_len: u32,
}

impl MatrixFileId {
    pub fn of_entry(entry: &ClientRegistryEntry) -> Self {
        Self {
            matrix_digest: entry.matrix_digest,
            file_root: entry.matrix_file_root,
            file_len: entry.matrix_file_len,
        }
    }

    /// The file a registration names.
    pub fn of_registration(
        registration: &jetsam_chain::consensus::client_objects::ClientRegistration,
    ) -> Self {
        Self {
            matrix_digest: registration.matrix_digest,
            file_root: registration.matrix_file_root,
            file_len: registration.matrix_file_len,
        }
    }

    pub fn chunk_count(&self) -> usize {
        matrix_file_chunk_count(self.file_len)
    }

    /// Exact length of chunk `index`, if the file has one.
    pub fn chunk_len(&self, index: u32) -> Option<usize> {
        let count = self.chunk_count();
        let index = usize::try_from(index).ok()?;
        if index >= count {
            return None;
        }
        if index + 1 == count {
            Some(self.file_len as usize - CLIENT_MATRIX_CHUNK_BYTES * (count - 1))
        } else {
            Some(CLIENT_MATRIX_CHUNK_BYTES)
        }
    }

    /// Exact length of the manifest: one digest per chunk.
    pub fn manifest_len(&self) -> usize {
        32 * self.chunk_count()
    }

    /// The chunk digests a manifest lists, if it is this file's.
    pub fn verify_manifest(&self, manifest: &[u8]) -> Option<Vec<Hash32>> {
        if manifest.len() != self.manifest_len() {
            return None;
        }
        let digests: Vec<Hash32> = manifest
            .chunks_exact(32)
            .map(|digest| digest.try_into().expect("32-byte chunks"))
            .collect();
        (matrix_file_root_from_chunk_digests(u64::from(self.file_len), &digests) == self.file_root)
            .then_some(digests)
    }

    /// The manifest of a file a node holds.
    pub fn manifest_of(file: &[u8]) -> Vec<u8> {
        file.chunks(CLIENT_MATRIX_CHUNK_BYTES)
            .enumerate()
            .flat_map(|(index, chunk)| matrix_file_chunk_digest(index as u32, chunk))
            .collect()
    }
}

fn check_proof_len(len: usize) -> Result<(), ClientTransportError> {
    if len > CLIENT_PROOF_MAX_WIRE_BYTES {
        return Err(ClientTransportError::ProofTooLarge {
            actual: len,
            max: CLIENT_PROOF_MAX_WIRE_BYTES,
        });
    }
    if len == 0 {
        return Err(ClientTransportError::EmptyProof);
    }
    Ok(())
}

fn take<'a>(src: &mut &'a [u8], n: usize) -> Result<&'a [u8], ClientTransportError> {
    if src.len() < n {
        return Err(ClientTransportError::Truncated);
    }
    let (head, tail) = src.split_at(n);
    *src = tail;
    Ok(head)
}

fn take_hash(src: &mut &[u8]) -> Result<Hash32, ClientTransportError> {
    Ok(take(src, 32)?.try_into().expect("32 bytes"))
}

fn take_u32(src: &mut &[u8]) -> Result<u32, ClientTransportError> {
    Ok(u32::from_le_bytes(
        take(src, 4)?.try_into().expect("4 bytes"),
    ))
}

/// Whether `chunk` is chunk `index` with this digest.
pub fn matrix_chunk_matches(index: u32, chunk: &[u8], digest: &Hash32) -> bool {
    matrix_file_chunk_digest(index, chunk) == *digest
}

// ---------------------------------------------------------------------------
// The requester's policy
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Wanted<P> {
    fee: u64,
    providers: Vec<P>,
    excluded: BTreeSet<P>,
    in_flight: Option<P>,
}

/// Which client proofs to fetch, from whom.
#[derive(Clone, Debug)]
pub struct ClientProofFetcher<P: Ord + Clone> {
    max_wanted: usize,
    max_providers: usize,
    max_in_flight_per_peer: usize,
    wanted: BTreeMap<ClientProofId, Wanted<P>>,
    in_flight: BTreeMap<P, usize>,
}

impl<P: Ord + Clone> ClientProofFetcher<P> {
    pub fn new(max_wanted: usize, max_providers: usize, max_in_flight_per_peer: usize) -> Self {
        Self {
            max_wanted: max_wanted.max(1),
            max_providers: max_providers.max(1),
            max_in_flight_per_peer: max_in_flight_per_peer.max(1),
            wanted: BTreeMap::new(),
            in_flight: BTreeMap::new(),
        }
    }

    /// Record that `peer` announced a proof. Returns whether it is wanted
    /// (bounded: past `max_wanted`, a better paid proof replaces the worst
    /// idle one).
    pub fn announced(&mut self, peer: P, announcement: ClientProofAnnouncement) -> bool {
        if let Some(wanted) = self.wanted.get_mut(&announcement.id) {
            if wanted.providers.contains(&peer) {
                return true;
            }
            if wanted.providers.len() >= self.max_providers {
                return false;
            }
            wanted.providers.push(peer);
            return true;
        }
        if self.wanted.len() >= self.max_wanted {
            let worst = self
                .wanted
                .iter()
                .filter(|(_, wanted)| wanted.in_flight.is_none())
                .min_by_key(|(id, wanted)| (wanted.fee, std::cmp::Reverse(**id)))
                .map(|(id, wanted)| (*id, wanted.fee));
            match worst {
                Some((id, fee)) if announcement.fee > fee => {
                    self.wanted.remove(&id);
                }
                _ => return false,
            }
        }
        self.wanted.insert(
            announcement.id,
            Wanted {
                fee: announcement.fee,
                providers: vec![peer],
                excluded: BTreeSet::new(),
                in_flight: None,
            },
        );
        true
    }

    /// Requests to send now: wanted proofs, best paid first, each to one
    /// provider neither excluded nor at its in-flight cap.
    pub fn next_requests(&mut self) -> Vec<(P, ClientProofId)> {
        let mut order: Vec<(u64, ClientProofId)> = self
            .wanted
            .iter()
            .filter(|(_, wanted)| wanted.in_flight.is_none())
            .map(|(id, wanted)| (wanted.fee, *id))
            .collect();
        order.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut requests = Vec::new();
        for (_, id) in order {
            let wanted = self.wanted.get_mut(&id).expect("listed above");
            let provider = wanted
                .providers
                .iter()
                .find(|peer| {
                    !wanted.excluded.contains(*peer)
                        && self.in_flight.get(*peer).copied().unwrap_or(0)
                            < self.max_in_flight_per_peer
                })
                .cloned();
            if let Some(peer) = provider {
                *self.in_flight.entry(peer.clone()).or_insert(0) += 1;
                wanted.in_flight = Some(peer.clone());
                requests.push((peer, id));
            }
        }
        requests
    }

    fn release(&mut self, peer: &P, id: &ClientProofId) -> bool {
        let Some(wanted) = self.wanted.get_mut(id) else {
            return false;
        };
        if wanted.in_flight.as_ref() != Some(peer) {
            return false;
        }
        wanted.in_flight = None;
        if let Some(count) = self.in_flight.get_mut(peer) {
            *count -= 1;
            if *count == 0 {
                self.in_flight.remove(peer);
            }
        }
        true
    }

    /// `peer` served exactly `id`'s bytes: the proof is fetched.
    pub fn completed(&mut self, peer: &P, id: &ClientProofId) {
        if self.release(peer, id) {
            self.wanted.remove(id);
        }
    }

    /// The request of `id` to `peer` failed. When `served_wrong_bytes`, the
    /// peer is excluded for this proof and the return value says to penalise
    /// it; a timeout or an unavailable answer only frees the request. The
    /// proof stays wanted while another provider remains.
    pub fn failed(&mut self, peer: &P, id: &ClientProofId, served_wrong_bytes: bool) -> bool {
        if !self.release(peer, id) {
            return false;
        }
        if served_wrong_bytes {
            let wanted = self.wanted.get_mut(id).expect("released above");
            wanted.excluded.insert(peer.clone());
            if wanted
                .providers
                .iter()
                .all(|provider| wanted.excluded.contains(provider))
            {
                self.wanted.remove(id);
            }
        }
        served_wrong_bytes
    }

    /// `peer` answered that it does not hold `id` (it relayed the
    /// announcement without fetching the bundle yet, or dropped it): it is
    /// not asked again for this proof and not reported. The proof is given
    /// up when no provider remains. Returns `false` (never a penalty).
    pub fn unavailable(&mut self, peer: &P, id: &ClientProofId) -> bool {
        if !self.release(peer, id) {
            return false;
        }
        self.exclude(peer, id);
        false
    }

    /// `peer` disconnected: its requests are freed, it leaves every proof's
    /// providers, and a proof no provider remains for is given up.
    pub fn forget_peer(&mut self, peer: &P) {
        let ids: Vec<ClientProofId> = self
            .wanted
            .iter()
            .filter(|(_, wanted)| wanted.providers.contains(peer))
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            self.release(peer, &id);
            self.exclude(peer, &id);
        }
        self.in_flight.remove(peer);
    }

    fn exclude(&mut self, peer: &P, id: &ClientProofId) {
        let Some(wanted) = self.wanted.get_mut(id) else {
            return;
        };
        wanted.excluded.insert(peer.clone());
        if wanted
            .providers
            .iter()
            .all(|provider| wanted.excluded.contains(provider))
        {
            self.wanted.remove(id);
        }
    }

    pub fn is_wanted(&self, id: &ClientProofId) -> bool {
        self.wanted.contains_key(id)
    }

    pub fn in_flight(&self, peer: &P) -> usize {
        self.in_flight.get(peer).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests;
