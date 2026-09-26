// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Plaintext key file: master_secret → disk.
//!
//! Format: `jetsam_plain_key_1` (16 bytes magic) + secret (32 bytes) = 48 bytes.
//!
//! Security model: the file is stored at `~/.jetsam/data/wallet.key` with
//! permissions 0o600 (owner-only). No encryption — the OS filesystem is the
//! security boundary during development. Future versions will derive the
//! master secret from a user-chosen file (photo, document, etc.) instead of
//! generating random bytes.

#[cfg(unix)]
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use rand::RngCore;
use zeroize::{Zeroize, ZeroizeOnDrop, Zeroizing};

use jetsam_poseidon2b::primitives::{Address, SpendSecret};

use super::encryption::{self, ENCRYPTED_FILE_LEN};

/// Environment variable holding the wallet passphrase.
///
/// JETSAM CHANGE: when it is set, the master secret is written encrypted with
/// Argon2id + XChaCha20-Poly1305 instead of in the clear. When it is not, the
/// cleartext format is still used and the caller is warned - refusing outright
/// would lock a developer out of an existing node for no gain.
pub const PASSPHRASE_ENV: &str = "JETSAM_WALLET_PASSPHRASE";

/// Test-only passphrase override.
///
/// Thread-local on purpose. The obvious alternative - having tests set and
/// unset the environment variable - mutates process-wide state while the other
/// tests run in parallel, which makes unrelated wallet tests fail at random.
/// A thread-local is read only by the thread that set it, so the cases stay
/// independent and the suite does not have to be pinned to one thread.
#[cfg(test)]
thread_local! {
    static PASSPHRASE_OVERRIDE: std::cell::RefCell<Option<Option<Vec<u8>>>> =
        const { std::cell::RefCell::new(None) };
}

/// Run `body` with the passphrase forced to `value` on this thread.
#[cfg(test)]
pub(super) fn with_passphrase<R>(value: Option<&[u8]>, body: impl FnOnce() -> R) -> R {
    PASSPHRASE_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = Some(value.map(<[u8]>::to_vec));
    });
    let out = body();
    PASSPHRASE_OVERRIDE.with(|cell| *cell.borrow_mut() = None);
    out
}

fn passphrase() -> Option<Zeroizing<Vec<u8>>> {
    #[cfg(test)]
    if let Some(override_value) = PASSPHRASE_OVERRIDE.with(|cell| cell.borrow().clone()) {
        return override_value.map(Zeroizing::new);
    }
    match std::env::var(PASSPHRASE_ENV) {
        Ok(value) if !value.is_empty() => Some(Zeroizing::new(value.into_bytes())),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum KeystoreError {
    #[error("wallet already exists at {0}")]
    AlreadyExists(PathBuf),
    #[error("wallet file not found at {0}")]
    NotFound(PathBuf),
    #[error("invalid wallet file format")]
    InvalidFormat,
    #[cfg(unix)]
    #[error("wallet file permissions are insecure: expected 0600, got {mode:04o}")]
    InsecurePermissions { mode: u32 },
    #[cfg(unix)]
    #[error("wallet file belongs to uid {actual}, expected current uid {expected}")]
    WrongOwner { actual: u32, expected: u32 },
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("wallet artifact: {0}")]
    Artifact(String),
    #[error("this wallet is encrypted; set {} to open it", PASSPHRASE_ENV)]
    PassphraseRequired,
    #[error("wallet encryption: {0}")]
    Encryption(#[from] super::encryption::EncryptionError),
}

// ---------------------------------------------------------------------------
// On-disk format (plaintext)
// ---------------------------------------------------------------------------

// JETSAM CHANGE: distinct on-disk magic so a Jetsam keystore can never be
// opened as an upstream one, or the reverse. Must stay exactly 16 bytes.
const PLAIN_MAGIC: &[u8; 16] = b"jetsam_plainkey1";

/// A wallet holding one spend secret supplied from outside, rather than a seed
/// it derives addresses from.
///
/// The development-fund addresses are built this way: an operator tool drew 32
/// bytes, called them the spend secret, and published `H(secret)` as the
/// address. There is no seed above them, so the ordinary keystore could not
/// hold them — and the coins at those addresses could not be moved at all.
/// Handing that secret to the ordinary import produces a valid, empty,
/// unrelated address, because the wallet hashes a seed once more to reach a
/// spend secret. A separate magic makes the two impossible to confuse.
const PLAIN_SPEND_MAGIC: &[u8; 16] = b"jetsam_spendkey1";

const SECRET_LEN: usize = 32;
const PLAIN_FILE_LEN: usize = 16 + SECRET_LEN; // 48 bytes

// ---------------------------------------------------------------------------
// MasterSecret
// ---------------------------------------------------------------------------

/// The decrypted master secret (zeroized on drop).
///
/// The field and the type are private to the wallet module. There is no raw
/// getter, formatter, clone, comparison, hash, or serialization surface.
#[derive(Zeroize, ZeroizeOnDrop)]
pub(super) struct MasterSecret([u8; SECRET_LEN]);

impl MasterSecret {
    /// Derive the spending secret for address index `n`.
    ///
    /// `spend_secret_n = Poseidon2b(master_secret, n, domain_tag)`
    ///
    /// The derived secret is used only by the wallet's witness-hiding authorization prover.
    /// It NEVER leaves the daemon.
    pub(super) fn derive_spend_secret(&self, index: u32) -> SpendSecret {
        use jetsam_core::{Block128, TowerField};
        use jetsam_poseidon2b::native::compression::Poseidon2bSponge;

        let mut sponge = Poseidon2bSponge::with_iv([Block128::ZERO; 2]);
        let mut master_fields = Zeroizing::new([
            Block128::from(u128::from_le_bytes(self.0[..16].try_into().unwrap())),
            Block128::from(u128::from_le_bytes(self.0[16..].try_into().unwrap())),
        ]);
        sponge.absorb(master_fields[0]);
        sponge.absorb(master_fields[1]);
        master_fields.zeroize();
        sponge.absorb(Block128::from(index as u128));
        sponge.absorb(Block128::from(0x6E6F69642D64657269_u128)); // "jetsam-deri"
        let mut digest = sponge.finalize();
        let secret = SpendSecret::from_bytes(digest);
        digest.zeroize();
        secret
    }

    /// Derive the public address for index `n` (safe to share).
    pub(super) fn derive_address(&self, index: u32) -> Address {
        let secret = self.derive_spend_secret(index);
        jetsam_poseidon2b::primitives::derive_address(&secret)
    }
}

// ---------------------------------------------------------------------------
// WalletSecret
// ---------------------------------------------------------------------------

/// What a wallet file holds, and therefore how many addresses it can own.
///
/// `Seed` is the ordinary wallet: one secret, an unlimited series of addresses
/// derived from it by index. `ImportedSpend` is one address and nothing more —
/// its spend secret was produced elsewhere, so there is no seed to walk.
pub(super) enum WalletSecret {
    Seed(MasterSecret),
    ImportedSpend(ImportedSpendSecret),
}

/// Which kind of secret a wallet file holds, for the export surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretKind {
    /// A seed the wallet derives every address from.
    Seed,
    /// One spend secret supplied from outside; one address, no more.
    ImportedSpend,
}

impl SecretKind {
    /// The flag that restores this secret. Naming the wrong one loses the
    /// address, so the export says it rather than leaving it to be guessed.
    pub fn restore_flag(self) -> &'static str {
        match self {
            Self::Seed => "--import-wallet-secret",
            Self::ImportedSpend => "--import-spend-secret",
        }
    }
}

/// A spend secret supplied from outside, held exactly as given.
#[derive(Zeroize, ZeroizeOnDrop)]
pub(super) struct ImportedSpendSecret([u8; SECRET_LEN]);

impl ImportedSpendSecret {
    fn spend_secret(&self) -> SpendSecret {
        SpendSecret::from_bytes(self.0)
    }
}

impl WalletSecret {
    /// True when this wallet owns exactly one address and cannot derive more.
    pub(super) fn holds_one_imported_address(&self) -> bool {
        matches!(self, Self::ImportedSpend(_))
    }

    /// The spending secret for address index `n`.
    ///
    /// An imported wallet has only index 0. Callers are held to that by
    /// `next_index`, which the loader pins to 1, so a higher index cannot be
    /// reached through the address list, the active-address switch or a send.
    pub(super) fn derive_spend_secret(&self, index: u32) -> SpendSecret {
        match self {
            Self::Seed(master) => master.derive_spend_secret(index),
            Self::ImportedSpend(imported) => imported.spend_secret(),
        }
    }

    /// The public address for index `n` (safe to share).
    pub(super) fn derive_address(&self, index: u32) -> Address {
        match self {
            Self::Seed(master) => master.derive_address(index),
            Self::ImportedSpend(imported) => {
                jetsam_poseidon2b::primitives::derive_address(&imported.spend_secret())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Keystore
// ---------------------------------------------------------------------------

/// Manages the plaintext wallet key file on disk.
pub(super) struct Keystore {
    path: PathBuf,
}

struct TemporaryKeyFile {
    path: PathBuf,
    armed: bool,
}

impl TemporaryKeyFile {
    fn new(path: PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn remove(mut self) -> std::io::Result<()> {
        std::fs::remove_file(&self.path)?;
        self.armed = false;
        Ok(())
    }
}

impl Drop for TemporaryKeyFile {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

impl Keystore {
    pub(super) fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub(super) fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Create a new wallet with a randomly generated secret.
    /// Writes `magic[16] + secret[32]` = 48 bytes with mode 0o600.
    /// Fails if the file already exists.
    pub(super) fn create_plain(&self) -> Result<MasterSecret, KeystoreError> {
        if self.exists() {
            return Err(KeystoreError::AlreadyExists(self.path.clone()));
        }
        let mut secret = Zeroizing::new([0u8; SECRET_LEN]);
        rand::thread_rng().fill_bytes(&mut *secret);

        // JETSAM CHANGE: encrypt when a passphrase is available.
        let buf = match passphrase() {
            Some(pass) => encryption::encode(&secret, &pass)?,
            None => {
                tracing::warn!(
                    "{PASSPHRASE_ENV} is not set: the wallet master secret is being written \
                     in cleartext. Anyone who can read this file takes the funds."
                );
                let mut plain = Zeroizing::new(Vec::with_capacity(PLAIN_FILE_LEN));
                plain.extend_from_slice(PLAIN_MAGIC);
                plain.extend_from_slice(&*secret);
                plain
            }
        };

        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Create a unique owner-only inode before the first secret byte is
        // written. `create_new` and O_NOFOLLOW reject a pre-existing temp path
        // or symlink. Linking that inode into the final name is atomic and,
        // unlike rename on Unix, cannot replace a wallet created concurrently.
        let tmp = self
            .path
            .with_extension(format!("tmp.{:032x}", rand::random::<u128>()));
        let temporary = TemporaryKeyFile::new(tmp.clone());
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options
                .mode(0o600)
                .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        }
        let mut file = options.open(&tmp)?;
        file.write_all(&buf)?;
        file.sync_all()?;
        drop(file);

        match std::fs::hard_link(&tmp, &self.path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(KeystoreError::AlreadyExists(self.path.clone()));
            }
            Err(error) => return Err(error.into()),
        }
        // The final link is already durable and safe to use. Temp cleanup is
        // best-effort: its guard retries once on failure, and any orphan keeps
        // the same owner-only mode rather than turning successful wallet
        // creation into a misleading externally-visible error.
        let _ = temporary.remove();

        #[cfg(unix)]
        if let Some(parent) = self.path.parent() {
            File::open(parent)?.sync_all()?;
        }

        let master = MasterSecret(*secret);
        secret.zeroize();
        Ok(master)
    }

    /// Load a wallet key file, whichever of the two kinds it is.
    pub(super) fn load_plain(&self) -> Result<WalletSecret, KeystoreError> {
        let data = self.read_plain_bytes()?;
        // JETSAM CHANGE: dispatch on the on-disk magic. Four shapes reach here
        // — seed or imported spend secret, each in cleartext or sealed — and
        // the magic is the only thing that tells them apart. Guessing is not an
        // option: both secrets are 32 opaque bytes, and the wrong guess spends
        // nothing while looking entirely healthy.
        let mut secret = Zeroizing::new([0u8; SECRET_LEN]);
        let imported = match encryption::encrypted_magic_of(&data) {
            Some(magic) => {
                let pass = passphrase().ok_or(KeystoreError::PassphraseRequired)?;
                secret.copy_from_slice(&*encryption::decode_as(magic, &data, &pass)?);
                magic == encryption::ENCRYPTED_SPEND_MAGIC
            }
            None => {
                secret.copy_from_slice(&data[16..]);
                &data[..16] == PLAIN_SPEND_MAGIC.as_ref()
            }
        };
        let loaded = if imported {
            WalletSecret::ImportedSpend(ImportedSpendSecret(*secret))
        } else {
            WalletSecret::Seed(MasterSecret(*secret))
        };
        secret.zeroize();
        Ok(loaded)
    }

    /// Export the 32 secret bytes as lowercase hexadecimal, and say which kind
    /// of secret they are.
    ///
    /// The kind is not decoration. The same 64 characters restore a wallet only
    /// through the matching import: fed to the other one they yield a valid,
    /// empty, unrelated address, and a backup that restores to the wrong
    /// address is not a backup. The on-disk magic stays an implementation
    /// detail; what it means does not.
    pub(super) fn export_secret_hex(
        &self,
    ) -> Result<(SecretKind, Zeroizing<String>), KeystoreError> {
        let data = self.read_plain_bytes()?;
        // JETSAM CHANGE: on an encrypted file the bytes after the magic are
        // ciphertext, not the secret. Exporting them would hand the user a
        // useless string and call it their key.
        if let Some(magic) = encryption::encrypted_magic_of(&data) {
            let pass = passphrase().ok_or(KeystoreError::PassphraseRequired)?;
            let secret = encryption::decode_as(magic, &data, &pass)?;
            let kind = if magic == encryption::ENCRYPTED_SPEND_MAGIC {
                SecretKind::ImportedSpend
            } else {
                SecretKind::Seed
            };
            return Ok((kind, Zeroizing::new(hex::encode(*secret))));
        }
        let kind = if &data[..16] == PLAIN_SPEND_MAGIC.as_ref() {
            SecretKind::ImportedSpend
        } else {
            SecretKind::Seed
        };
        Ok((kind, Zeroizing::new(hex::encode(&data[PLAIN_MAGIC.len()..]))))
    }

    /// Build the private on-disk key artifact from one validated master
    /// secret. Callers keep the returned bytes zeroized and install them
    /// through the wallet's atomic replacement path.
    pub(super) fn encode_plain_file(master_secret: &[u8; SECRET_LEN]) -> Zeroizing<Vec<u8>> {
        let mut encoded = Zeroizing::new(Vec::with_capacity(PLAIN_FILE_LEN));
        encoded.extend_from_slice(PLAIN_MAGIC);
        encoded.extend_from_slice(master_secret);
        encoded
    }

    /// The same artifact for a spend secret supplied from outside, sealed under
    /// the passphrase when one is set.
    ///
    /// An imported secret is a treasury key far more often than a pocket one:
    /// writing it in the clear on a machine whose operator asked for a
    /// passphrase would be the wrong default to choose in new code. Encryption
    /// failing is an error and not a reason to fall back to cleartext.
    pub(super) fn encode_imported_spend_file(
        spend_secret: &[u8; SECRET_LEN],
    ) -> Result<Zeroizing<Vec<u8>>, KeystoreError> {
        if let Some(pass) = passphrase() {
            return Ok(encryption::encode_as(
                encryption::ENCRYPTED_SPEND_MAGIC,
                spend_secret,
                &pass,
            )?);
        }
        tracing::warn!(
            "{PASSPHRASE_ENV} is not set: this imported spend secret is being written \
             in cleartext. Anyone who can read this file takes the funds."
        );
        let mut encoded = Zeroizing::new(Vec::with_capacity(PLAIN_FILE_LEN));
        encoded.extend_from_slice(PLAIN_SPEND_MAGIC);
        encoded.extend_from_slice(spend_secret);
        Ok(encoded)
    }

    fn read_plain_bytes(&self) -> Result<Zeroizing<Vec<u8>>, KeystoreError> {
        if !self.exists() {
            return Err(KeystoreError::NotFound(self.path.clone()));
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        }
        let file = options.open(&self.path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(KeystoreError::InvalidFormat);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            let mode = metadata.permissions().mode() & 0o777;
            if mode != 0o600 {
                return Err(KeystoreError::InsecurePermissions { mode });
            }
            let actual = metadata.uid();
            // SAFETY: `geteuid` has no preconditions and does not dereference
            // pointers or mutate process state.
            let expected = unsafe { libc::geteuid() };
            if actual != expected {
                return Err(KeystoreError::WrongOwner { actual, expected });
            }
        }
        // JETSAM CHANGE: accept BOTH the cleartext format and the encrypted one.
        // An existing wallet must never become unreadable because the format
        // moved on - losing a keystore loses the coins for good.
        let cap = ENCRYPTED_FILE_LEN.max(PLAIN_FILE_LEN) + 1;
        let mut data = Zeroizing::new(Vec::with_capacity(cap));
        file.take(cap as u64).read_to_end(&mut data)?;
        if super::encryption::is_encrypted(&data) {
            if data.len() != ENCRYPTED_FILE_LEN {
                return Err(KeystoreError::InvalidFormat);
            }
            return Ok(data);
        }
        if data.len() != PLAIN_FILE_LEN {
            return Err(KeystoreError::InvalidFormat);
        }
        if &data[..16] != PLAIN_MAGIC.as_ref() && &data[..16] != PLAIN_SPEND_MAGIC.as_ref() {
            return Err(KeystoreError::InvalidFormat);
        }
        Ok(data)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// JETSAM — the wallet round-trips through the encrypted format, refuses to
    /// open without the passphrase, and fails closed on a wrong one.
    #[test]
    fn create_and_load_encrypted() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("wallet.key");
        let ks = Keystore::new(path.clone());

        let created = with_passphrase(Some(b"a real passphrase"), || ks.create_plain().unwrap());

        let raw = std::fs::read(&path).unwrap();
        assert!(
            super::encryption::is_encrypted(&raw),
            "a passphrase was set, so the file must be encrypted"
        );

        let loaded = with_passphrase(Some(b"a real passphrase"), || ks.load_plain().unwrap());
        assert_eq!(created.derive_address(0), loaded.derive_address(0));
        assert_eq!(created.derive_address(99), loaded.derive_address(99));

        // No passphrase: refuse, rather than crash or return something wrong.
        assert!(matches!(
            with_passphrase(None, || ks.load_plain()),
            Err(KeystoreError::PassphraseRequired)
        ));

        // Wrong passphrase: fail closed.
        assert!(with_passphrase(Some(b"the wrong passphrase"), || ks.load_plain()).is_err());
    }

    /// A cleartext wallet written before encryption existed must still open,
    /// with or without a passphrase set. Losing a keystore loses the coins.
    #[test]
    fn cleartext_wallet_still_opens() {
        let dir = TempDir::new().unwrap();
        let ks = Keystore::new(dir.path().join("wallet.key"));
        let created = with_passphrase(None, || ks.create_plain().unwrap());
        for pass in [None, Some(b"irrelevant".as_slice())] {
            let loaded = with_passphrase(pass, || ks.load_plain().unwrap());
            assert_eq!(created.derive_address(0), loaded.derive_address(0));
        }
    }

    #[test]
    fn create_and_load_plain() {
        let dir = TempDir::new().unwrap();
        let ks = Keystore::new(dir.path().join("wallet.key"));
        let secret = ks.create_plain().unwrap();
        let loaded = ks.load_plain().unwrap();
        assert_eq!(secret.derive_address(0), loaded.derive_address(0));
        assert_eq!(secret.derive_address(99), loaded.derive_address(99));
    }

    /// 32 bytes drawn by an operator tool, the way the development-fund
    /// addresses were made.
    const RAW_SPEND: [u8; SECRET_LEN] = [0x2C; SECRET_LEN];

    /// The address that secret is published as: one hash, no seed above it.
    fn published_address(raw: [u8; SECRET_LEN]) -> Address {
        jetsam_poseidon2b::primitives::derive_address(&SpendSecret::from_bytes(raw))
    }

    fn write_wallet(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
    }

    /// The whole point: a wallet carrying an imported spend secret owns the
    /// address that secret was published as. Before this existed, the coins on
    /// the development-fund addresses could not be moved by any binary we ship.
    #[test]
    fn an_imported_spend_secret_owns_the_address_it_was_published_as() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("wallet.key");
        let ks = Keystore::new(&path);
        write_wallet(
            &path,
            &with_passphrase(None, || {
                Keystore::encode_imported_spend_file(&RAW_SPEND).unwrap()
            }),
        );

        let loaded = with_passphrase(None, || ks.load_plain().unwrap());
        assert!(loaded.holds_one_imported_address());
        assert_eq!(loaded.derive_address(0), published_address(RAW_SPEND));
    }

    /// The trap that cost an evening: the same 64 characters restored the wrong
    /// way give a valid, empty, entirely unrelated address. Nothing warns you —
    /// which is why the two files carry different magics.
    #[test]
    fn the_same_bytes_as_a_seed_reach_a_different_address() {
        let dir = TempDir::new().unwrap();
        let as_seed = dir.path().join("seed.key");
        let as_spend = dir.path().join("spend.key");
        write_wallet(&as_seed, &Keystore::encode_plain_file(&RAW_SPEND));
        write_wallet(
            &as_spend,
            &with_passphrase(None, || {
                Keystore::encode_imported_spend_file(&RAW_SPEND).unwrap()
            }),
        );

        let seed_address = with_passphrase(None, || {
            Keystore::new(&as_seed).load_plain().unwrap().derive_address(0)
        });
        let spend_address = with_passphrase(None, || {
            Keystore::new(&as_spend)
                .load_plain()
                .unwrap()
                .derive_address(0)
        });

        assert_eq!(spend_address, published_address(RAW_SPEND));
        assert_ne!(
            seed_address, spend_address,
            "reading a spend secret as a seed must not silently land on the right address"
        );
    }

    /// An imported secret is a treasury key more often than a pocket one, so it
    /// is sealed wherever the ordinary wallet would be — and it round-trips.
    #[test]
    fn an_imported_spend_secret_round_trips_through_the_sealed_format() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("wallet.key");
        let ks = Keystore::new(&path);
        write_wallet(
            &path,
            &with_passphrase(Some(b"a real passphrase"), || {
                Keystore::encode_imported_spend_file(&RAW_SPEND).unwrap()
            }),
        );
        assert!(
            super::encryption::is_encrypted(&std::fs::read(&path).unwrap()),
            "a passphrase was set, so the file must be encrypted"
        );

        let loaded = with_passphrase(Some(b"a real passphrase"), || ks.load_plain().unwrap());
        assert!(loaded.holds_one_imported_address());
        assert_eq!(loaded.derive_address(0), published_address(RAW_SPEND));

        assert!(matches!(
            with_passphrase(None, || ks.load_plain()),
            Err(KeystoreError::PassphraseRequired)
        ));
    }

    /// A backup is only a backup if it says how to restore it.
    #[test]
    fn the_export_names_the_import_that_restores_it() {
        let dir = TempDir::new().unwrap();
        let seed_path = dir.path().join("seed.key");
        let spend_path = dir.path().join("spend.key");
        write_wallet(&seed_path, &Keystore::encode_plain_file(&RAW_SPEND));
        write_wallet(
            &spend_path,
            &with_passphrase(None, || {
                Keystore::encode_imported_spend_file(&RAW_SPEND).unwrap()
            }),
        );

        let expected = hex::encode(RAW_SPEND);
        for (path, kind, flag) in [
            (&seed_path, SecretKind::Seed, "--import-wallet-secret"),
            (
                &spend_path,
                SecretKind::ImportedSpend,
                "--import-spend-secret",
            ),
        ] {
            let (found, hex_secret) =
                with_passphrase(None, || Keystore::new(path).export_secret_hex().unwrap());
            assert_eq!(found, kind);
            assert_eq!(hex_secret.as_str(), expected);
            assert_eq!(found.restore_flag(), flag);
        }
    }

    #[test]
    fn double_create_fails() {
        let dir = TempDir::new().unwrap();
        let ks = Keystore::new(dir.path().join("wallet.key"));
        ks.create_plain().unwrap();
        assert!(matches!(
            ks.create_plain(),
            Err(KeystoreError::AlreadyExists(_))
        ));
    }

    #[test]
    fn load_missing_fails() {
        let dir = TempDir::new().unwrap();
        let ks = Keystore::new(dir.path().join("wallet.key"));
        assert!(matches!(ks.load_plain(), Err(KeystoreError::NotFound(_))));
    }

    #[test]
    fn address_derivation_deterministic() {
        let dir = TempDir::new().unwrap();
        let ks = Keystore::new(dir.path().join("wallet.key"));
        let secret = ks.create_plain().unwrap();
        assert_eq!(secret.derive_address(0), secret.derive_address(0));
        assert_ne!(secret.derive_address(0), secret.derive_address(1));
    }

    #[test]
    fn spend_secret_differs_per_index() {
        let dir = TempDir::new().unwrap();
        let ks = Keystore::new(dir.path().join("wallet.key"));
        let master = ks.create_plain().unwrap();
        assert_ne!(master.derive_address(0), master.derive_address(1));
    }

    #[test]
    fn master_secret_traits_and_explicit_zeroize_are_pinned() {
        static_assertions::assert_not_impl_any!(
            MasterSecret: Copy,
            Clone,
            std::fmt::Debug,
            PartialEq,
            Eq,
            std::hash::Hash,
            serde::Serialize,
            serde::de::DeserializeOwned
        );
        fn assert_zeroize<T: Zeroize + ZeroizeOnDrop>() {}
        assert_zeroize::<MasterSecret>();

        let mut secret = MasterSecret([0xA6; SECRET_LEN]);
        secret.zeroize();
        assert_eq!(secret.0, [0u8; SECRET_LEN]);
    }

    #[cfg(unix)]
    #[test]
    fn key_is_private_before_and_after_load() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("wallet.key");
        let ks = Keystore::new(&path);
        ks.create_plain().unwrap();

        let metadata = std::fs::symlink_metadata(&path).unwrap();
        assert!(metadata.file_type().is_file());
        assert_eq!(metadata.permissions().mode() & 0o777, 0o600);
        assert_eq!(metadata.uid(), unsafe { libc::geteuid() });
        ks.load_plain().unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn load_rejects_public_permissions_and_symlinks() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("wallet.key");
        let ks = Keystore::new(&path);
        ks.create_plain().unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            ks.load_plain(),
            Err(KeystoreError::InsecurePermissions { mode: 0o644 })
        ));

        let link = dir.path().join("wallet-link.key");
        symlink(&path, &link).unwrap();
        let linked = Keystore::new(link);
        assert!(matches!(linked.load_plain(), Err(KeystoreError::Io(_))));
    }
}
