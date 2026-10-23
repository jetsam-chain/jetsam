// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.6 batch proofs between nodes: what an aggregator hands to
//! miners, the way a v1.5 client proof travels.
//!
//! A batch proof is bounded by the class's encoding
//! ([`CLIENT_BATCH_PROOF_MAX_WIRE_BYTES`], v1.6 only: the v1.5 client bound is
//! unchanged). A node that holds a batch **announces** it in a few dozen
//! bytes ([`ClientBatchAnnouncement`]: the batch submission, the bundle's
//! byte digest and length, the fee); a node that wants it requests the
//! [`ClientBatchBundle`] (batch submission, paying transaction, batch proof)
//! by that id. The response is refused unless its length and byte digest are
//! exactly the id's; the bundle is decoded under its bounds (a batch proof
//! longer than the class's bound is refused before it is read) and its
//! payment must name its batch submission. The miner then decides the batch
//! natively against its tip's lanes (`jetsam_recursive`).

use jetsam_chain::consensus::client_objects::{
    pays_batch_submission, ClientBatchSubmission, CLIENT_BATCH_PROOF_MAX_WIRE_BYTES,
};
use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_tx::{PagedSpendIntent, MAX_PAGED_SPEND_INTENT_BYTES};

use super::{take, take_hash, take_u32, ClientTransportError, Hash32};

const BATCH_BUNDLE_MAGIC: [u8; 4] = *b"JCB1";
const BATCH_ANNOUNCEMENT_MAGIC: [u8; 4] = *b"JBA1";
const BATCH_BUNDLE_DIGEST_DOMAIN: &[u8] = b"JTM/P2P/CLIENT-BATCH-BUNDLE/V1";

/// Fixed bytes of a batch bundle: magic, batch submission (class, root,
/// count), the two length prefixes.
pub const CLIENT_BATCH_BUNDLE_FIXED_BYTES: usize = 4 + 32 + 32 + 4 + 4 + 4;

/// Largest batch bundle: one maximal payment intent and one maximal batch
/// proof.
pub const MAX_CLIENT_BATCH_BUNDLE_BYTES: usize =
    CLIENT_BATCH_BUNDLE_FIXED_BYTES + MAX_PAGED_SPEND_INTENT_BYTES + CLIENT_BATCH_PROOF_MAX_WIRE_BYTES;

/// Exact size of a batch announcement.
pub const CLIENT_BATCH_ANNOUNCEMENT_BYTES: usize = 4 + 32 + 32 + 4 + 32 + 4 + 8;

fn check_batch_proof_len(len: usize) -> Result<(), ClientTransportError> {
    if len > CLIENT_BATCH_PROOF_MAX_WIRE_BYTES {
        return Err(ClientTransportError::ProofTooLarge {
            actual: len,
            max: CLIENT_BATCH_PROOF_MAX_WIRE_BYTES,
        });
    }
    if len == 0 {
        return Err(ClientTransportError::EmptyProof);
    }
    Ok(())
}

fn put_submission(bytes: &mut Vec<u8>, submission: &ClientBatchSubmission) {
    bytes.extend_from_slice(&submission.batch_class);
    bytes.extend_from_slice(&submission.statements_root);
    bytes.extend_from_slice(&submission.count.to_le_bytes());
}

fn take_submission(src: &mut &[u8]) -> Result<ClientBatchSubmission, ClientTransportError> {
    Ok(ClientBatchSubmission {
        batch_class: take_hash(src)?,
        statements_root: take_hash(src)?,
        count: take_u32(src)?,
    })
}

/// Network identity of one batch bundle: its batch submission and the digest
/// and length of its exact bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClientBatchId {
    pub submission: ClientBatchSubmission,
    pub bundle_digest: Hash32,
    pub encoded_len: u32,
}

impl ClientBatchId {
    pub fn of_bytes(submission: ClientBatchSubmission, bytes: &[u8]) -> Option<Self> {
        Some(Self {
            submission,
            bundle_digest: poseidon2b_hash_bytes(BATCH_BUNDLE_DIGEST_DOMAIN, bytes),
            encoded_len: u32::try_from(bytes.len()).ok()?,
        })
    }

    pub fn matches_bytes(&self, bytes: &[u8]) -> bool {
        usize::try_from(self.encoded_len).ok() == Some(bytes.len())
            && self.bundle_digest == poseidon2b_hash_bytes(BATCH_BUNDLE_DIGEST_DOMAIN, bytes)
    }
}

/// What travels for one batch submission: the submission, the ordinary
/// transaction that pays it, and the batch proof (opaque here).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClientBatchBundle {
    pub submission: ClientBatchSubmission,
    pub payment: PagedSpendIntent,
    pub proof_bytes: Vec<u8>,
}

impl ClientBatchBundle {
    pub fn new(
        submission: ClientBatchSubmission,
        payment: PagedSpendIntent,
        proof_bytes: Vec<u8>,
    ) -> Result<Self, ClientTransportError> {
        check_batch_proof_len(proof_bytes.len())?;
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
        if !pays_batch_submission(&payment.pages, &submission) {
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
            CLIENT_BATCH_BUNDLE_FIXED_BYTES + payment.len() + self.proof_bytes.len(),
        );
        bytes.extend_from_slice(&BATCH_BUNDLE_MAGIC);
        put_submission(&mut bytes, &self.submission);
        bytes.extend_from_slice(&(payment.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payment);
        bytes.extend_from_slice(&(self.proof_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&self.proof_bytes);
        bytes
    }

    /// Decode a batch bundle, every length bounded before it is read.
    pub fn decode(bytes: &[u8]) -> Result<Self, ClientTransportError> {
        let mut src = bytes;
        if take(&mut src, 4)? != BATCH_BUNDLE_MAGIC {
            return Err(ClientTransportError::BadMagic);
        }
        let submission = take_submission(&mut src)?;
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
        check_batch_proof_len(proof_len)?;
        let proof_bytes = take(&mut src, proof_len)?.to_vec();
        if !src.is_empty() {
            return Err(ClientTransportError::TrailingBytes);
        }
        if !pays_batch_submission(&payment.pages, &submission) {
            return Err(ClientTransportError::NotThePayment);
        }
        Ok(Self {
            submission,
            payment,
            proof_bytes,
        })
    }

    /// Decode the bytes received for `id`: exactly its bytes, then exactly its
    /// batch submission.
    pub fn decode_for(id: &ClientBatchId, bytes: &[u8]) -> Result<Self, ClientTransportError> {
        if !id.matches_bytes(bytes) {
            return Err(ClientTransportError::WrongBytes);
        }
        let bundle = Self::decode(bytes)?;
        if bundle.submission != id.submission {
            return Err(ClientTransportError::NotThePayment);
        }
        Ok(bundle)
    }

    pub fn id(&self) -> ClientBatchId {
        let bytes = self.encode();
        ClientBatchId::of_bytes(self.submission, &bytes)
            .expect("a bundle is bounded below u32::MAX bytes")
    }
}

/// Gossip-sized notice that a node holds one batch bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClientBatchAnnouncement {
    pub id: ClientBatchId,
    /// The paying transaction's fee, the miner's ranking key.
    pub fee: u64,
}

impl ClientBatchAnnouncement {
    pub fn encode(&self) -> [u8; CLIENT_BATCH_ANNOUNCEMENT_BYTES] {
        let mut bytes = Vec::with_capacity(CLIENT_BATCH_ANNOUNCEMENT_BYTES);
        bytes.extend_from_slice(&BATCH_ANNOUNCEMENT_MAGIC);
        put_submission(&mut bytes, &self.id.submission);
        bytes.extend_from_slice(&self.id.bundle_digest);
        bytes.extend_from_slice(&self.id.encoded_len.to_le_bytes());
        bytes.extend_from_slice(&self.fee.to_le_bytes());
        bytes.try_into().expect("a batch announcement has a fixed size")
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, ClientTransportError> {
        match bytes.len().cmp(&CLIENT_BATCH_ANNOUNCEMENT_BYTES) {
            std::cmp::Ordering::Less => return Err(ClientTransportError::Truncated),
            std::cmp::Ordering::Greater => return Err(ClientTransportError::TrailingBytes),
            std::cmp::Ordering::Equal => {}
        }
        let mut src = bytes;
        if take(&mut src, 4)? != BATCH_ANNOUNCEMENT_MAGIC {
            return Err(ClientTransportError::BadMagic);
        }
        let submission = take_submission(&mut src)?;
        let bundle_digest = take_hash(&mut src)?;
        let encoded_len = take_u32(&mut src)?;
        let fee = u64::from_le_bytes(take(&mut src, 8)?.try_into().expect("8 bytes"));
        Ok(Self {
            id: ClientBatchId {
                submission,
                bundle_digest,
                encoded_len,
            },
            fee,
        })
    }
}

#[cfg(test)]
mod tests;
