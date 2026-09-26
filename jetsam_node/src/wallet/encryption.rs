// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Passphrase encryption for the wallet master secret.
//!
//! # Why this exists
//!
//! The keystore this fork started from stored the master secret in cleartext
//! and said so: owner-only file permissions were the entire security boundary.
//! Anyone who can read one file on a user's machine — a stray backup, a synced
//! folder, a second process running as the same user, a stolen disk — takes
//! every coin that wallet holds. That will cost more coins than any
//! cryptanalytic weakness in the chain ever does.
//!
//! # Construction
//!
//! Deliberately boring, and assembled only from reviewed primitives:
//!
//! - **Argon2id** turns the passphrase into a 32-byte key. Memory-hard, so a
//!   stolen file cannot be attacked at GPU speed. 64 MiB, 3 passes, 1 lane.
//! - **XChaCha20-Poly1305** encrypts the secret under that key. The 24-byte
//!   nonce is random per write, which at that width makes reuse a non-issue.
//! - The header — magic, version, salt, nonce — is **authenticated as
//!   associated data**, so downgrading the version or swapping the salt of a
//!   captured file is rejected rather than silently honoured.
//!
//! Nothing here is hand-rolled. That is the point: a bespoke construction
//! would be a worse outcome than the cleartext file it replaces.
//!
//! # On-disk layout (105 bytes)
//!
//! ```text
//! magic       16  "jetsam_aead_key1" for a seed,
//!                 "jetsam_aeadspnd1" for an imported spend secret
//! version      1  0x01
//! salt        16  Argon2id salt
//! nonce       24  XChaCha20-Poly1305 nonce
//! ciphertext  48  32-byte secret + 16-byte Poly1305 tag
//! ```

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{XChaCha20Poly1305, XNonce};
use rand::RngCore;
use zeroize::Zeroizing;

/// Distinct from the cleartext magic, so `load` can tell the two apart without
/// guessing and an upstream file can never be read as one of ours.
pub(super) const ENCRYPTED_MAGIC: &[u8; 16] = b"jetsam_aead_key1";

/// The same container, holding a spend secret supplied from outside rather than
/// a master seed.
///
/// The two are both 32 random-looking bytes and mean entirely different things:
/// a master seed is hashed once more to reach a spend secret, so reading one as
/// the other silently produces a valid, empty, unrelated address. The magic is
/// part of the authenticated header, so this is not a label anyone can ignore —
/// a file sealed under one magic cannot be opened under the other at all.
pub(super) const ENCRYPTED_SPEND_MAGIC: &[u8; 16] = b"jetsam_aeadspnd1";

pub(super) const FORMAT_VERSION: u8 = 1;

const SECRET_LEN: usize = 32;
const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 24;
const TAG_LEN: usize = 16;
const HEADER_LEN: usize = ENCRYPTED_MAGIC.len() + 1 + SALT_LEN + NONCE_LEN;
pub(super) const ENCRYPTED_FILE_LEN: usize = HEADER_LEN + SECRET_LEN + TAG_LEN;

/// Argon2id cost. 64 MiB and three passes is deliberately painful for an
/// offline attacker and unnoticeable once, at wallet open.
const MEMORY_KIB: u32 = 64 * 1024;
const ITERATIONS: u32 = 3;
const LANES: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum EncryptionError {
    #[error("wallet file is not in the encrypted format")]
    NotEncrypted,
    #[error("unsupported encrypted wallet version {0}")]
    UnsupportedVersion(u8),
    #[error("encrypted wallet file is truncated or corrupt")]
    Malformed,
    #[error("wrong passphrase, or the wallet file has been tampered with")]
    Authentication,
    #[error("key derivation failed: {0}")]
    KeyDerivation(String),
}

/// True when `bytes` begins with either encrypted-format magic.
pub(super) fn is_encrypted(bytes: &[u8]) -> bool {
    encrypted_magic_of(bytes).is_some()
}

/// Which of the two encrypted containers `bytes` is, if either.
pub(super) fn encrypted_magic_of(bytes: &[u8]) -> Option<&'static [u8; 16]> {
    for magic in [ENCRYPTED_MAGIC, ENCRYPTED_SPEND_MAGIC] {
        if bytes.len() >= magic.len() && &bytes[..magic.len()] == magic {
            return Some(magic);
        }
    }
    None
}

fn derive_key(
    passphrase: &[u8],
    salt: &[u8; SALT_LEN],
) -> Result<Zeroizing<[u8; 32]>, EncryptionError> {
    let params = Params::new(MEMORY_KIB, ITERATIONS, LANES, Some(32))
        .map_err(|error| EncryptionError::KeyDerivation(error.to_string()))?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = Zeroizing::new([0u8; 32]);
    argon
        .hash_password_into(passphrase, salt, &mut *key)
        .map_err(|error| EncryptionError::KeyDerivation(error.to_string()))?;
    Ok(key)
}

/// Header bytes, which are authenticated but not encrypted.
fn header(magic: &[u8; 16], salt: &[u8; SALT_LEN], nonce: &[u8; NONCE_LEN]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN);
    out.extend_from_slice(magic);
    out.push(FORMAT_VERSION);
    out.extend_from_slice(salt);
    out.extend_from_slice(nonce);
    out
}

/// Encrypt one master secret under `passphrase`.
pub(super) fn encode(
    secret: &[u8; SECRET_LEN],
    passphrase: &[u8],
) -> Result<Zeroizing<Vec<u8>>, EncryptionError> {
    encode_as(ENCRYPTED_MAGIC, secret, passphrase)
}

/// Encrypt 32 secret bytes under `passphrase`, sealed as `magic`.
pub(super) fn encode_as(
    magic: &[u8; 16],
    secret: &[u8; SECRET_LEN],
    passphrase: &[u8],
) -> Result<Zeroizing<Vec<u8>>, EncryptionError> {
    let mut salt = [0u8; SALT_LEN];
    let mut nonce = [0u8; NONCE_LEN];
    rand::thread_rng().fill_bytes(&mut salt);
    rand::thread_rng().fill_bytes(&mut nonce);

    let key = derive_key(passphrase, &salt)?;
    let cipher = XChaCha20Poly1305::new(key.as_slice().into());
    let aad = header(magic, &salt, &nonce);
    let sealed = cipher
        .encrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: secret.as_slice(),
                aad: &aad,
            },
        )
        .map_err(|_| EncryptionError::Authentication)?;

    let mut out = Zeroizing::new(Vec::with_capacity(ENCRYPTED_FILE_LEN));
    out.extend_from_slice(&aad);
    out.extend_from_slice(&sealed);
    debug_assert_eq!(out.len(), ENCRYPTED_FILE_LEN);
    Ok(out)
}

/// Decrypt 32 secret bytes sealed as `magic`. Fails closed on a wrong
/// passphrase, on any modification of the file, header included, and on a file
/// that was sealed as the other kind of secret.
pub(super) fn decode_as(
    magic: &[u8; 16],
    bytes: &[u8],
    passphrase: &[u8],
) -> Result<Zeroizing<[u8; SECRET_LEN]>, EncryptionError> {
    if encrypted_magic_of(bytes) != Some(magic) {
        return Err(EncryptionError::NotEncrypted);
    }
    if bytes.len() != ENCRYPTED_FILE_LEN {
        return Err(EncryptionError::Malformed);
    }
    let version = bytes[magic.len()];
    if version != FORMAT_VERSION {
        return Err(EncryptionError::UnsupportedVersion(version));
    }

    let salt_at = magic.len() + 1;
    let nonce_at = salt_at + SALT_LEN;
    let mut salt = [0u8; SALT_LEN];
    let mut nonce = [0u8; NONCE_LEN];
    salt.copy_from_slice(&bytes[salt_at..nonce_at]);
    nonce.copy_from_slice(&bytes[nonce_at..HEADER_LEN]);

    let key = derive_key(passphrase, &salt)?;
    let cipher = XChaCha20Poly1305::new(key.as_slice().into());
    let opened = cipher
        .decrypt(
            XNonce::from_slice(&nonce),
            Payload {
                msg: &bytes[HEADER_LEN..],
                aad: &bytes[..HEADER_LEN],
            },
        )
        .map_err(|_| EncryptionError::Authentication)?;

    if opened.len() != SECRET_LEN {
        return Err(EncryptionError::Malformed);
    }
    let mut secret = Zeroizing::new([0u8; SECRET_LEN]);
    secret.copy_from_slice(&opened);
    Ok(secret)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: [u8; SECRET_LEN] = [0x5A; SECRET_LEN];
    const PASS: &[u8] = b"correct horse battery staple";

    /// A master seed and a spend secret are both 32 opaque bytes, and reading
    /// one as the other yields a valid address holding nothing. The magic sits
    /// in the authenticated header precisely so that mistake cannot be made
    /// with the right passphrase in hand.
    #[test]
    fn a_spend_secret_cannot_be_opened_as_a_master_seed() {
        let sealed = encode_as(ENCRYPTED_SPEND_MAGIC, &SECRET, PASS).unwrap();
        assert!(matches!(
            decode_as(ENCRYPTED_MAGIC, &sealed, PASS),
            Err(EncryptionError::NotEncrypted)
        ));
        assert_eq!(*decode_as(ENCRYPTED_SPEND_MAGIC, &sealed, PASS).unwrap(), SECRET);
    }

    #[test]
    fn a_master_seed_cannot_be_opened_as_a_spend_secret() {
        let sealed = encode(&SECRET, PASS).unwrap();
        assert!(matches!(
            decode_as(ENCRYPTED_SPEND_MAGIC, &sealed, PASS),
            Err(EncryptionError::NotEncrypted)
        ));
        assert_eq!(*decode_as(ENCRYPTED_MAGIC, &sealed, PASS).unwrap(), SECRET);
    }

    #[test]
    fn round_trip_recovers_the_secret() {
        let sealed = encode(&SECRET, PASS).expect("encrypt");
        assert_eq!(sealed.len(), ENCRYPTED_FILE_LEN);
        assert!(is_encrypted(&sealed));
        let opened = decode_as(ENCRYPTED_MAGIC, &sealed, PASS).expect("decrypt");
        assert_eq!(*opened, SECRET);
    }

    /// The secret must not appear anywhere in the file.
    #[test]
    fn ciphertext_does_not_contain_the_plaintext() {
        let sealed = encode(&SECRET, PASS).expect("encrypt");
        assert!(
            !sealed.windows(SECRET_LEN).any(|w| w == SECRET),
            "the master secret is present in cleartext inside the file"
        );
    }

    #[test]
    fn wrong_passphrase_is_rejected() {
        let sealed = encode(&SECRET, PASS).expect("encrypt");
        assert!(matches!(
            decode_as(ENCRYPTED_MAGIC, &sealed, b"not the passphrase"),
            Err(EncryptionError::Authentication)
        ));
    }

    /// Every region is authenticated — header included, which is what stops a
    /// captured file from being downgraded or re-salted.
    ///
    /// One offset per structural region rather than all 89: each Argon2id
    /// derivation costs 64 MiB by design, so an exhaustive sweep would make
    /// this the slowest test in the tree for no extra coverage.
    #[test]
    fn a_bit_flip_in_any_region_is_rejected() {
        let sealed = encode(&SECRET, PASS).expect("encrypt");
        let salt_at = ENCRYPTED_MAGIC.len() + 1;
        let nonce_at = salt_at + SALT_LEN;
        let regions = [
            ("version", ENCRYPTED_MAGIC.len()),
            ("salt", salt_at + SALT_LEN / 2),
            ("nonce", nonce_at + NONCE_LEN / 2),
            ("ciphertext", HEADER_LEN + SECRET_LEN / 2),
            ("tag", sealed.len() - 1),
        ];
        for (region, index) in regions {
            let mut tampered = sealed.to_vec();
            tampered[index] ^= 0x01;
            assert!(
                decode_as(ENCRYPTED_MAGIC, &tampered, PASS).is_err(),
                "a flipped bit in the {region} at offset {index} was accepted"
            );
        }
    }

    #[test]
    fn two_writes_of_one_secret_differ() {
        let a = encode(&SECRET, PASS).expect("encrypt");
        let b = encode(&SECRET, PASS).expect("encrypt");
        assert_ne!(*a, *b, "salt and nonce must be fresh on every write");
    }

    #[test]
    fn a_cleartext_file_is_not_mistaken_for_an_encrypted_one() {
        let mut plain = Vec::from(*b"jetsam_plainkey1");
        plain.extend_from_slice(&SECRET);
        assert!(!is_encrypted(&plain));
        assert!(matches!(
            decode_as(ENCRYPTED_MAGIC, &plain, PASS),
            Err(EncryptionError::NotEncrypted)
        ));
    }
}
