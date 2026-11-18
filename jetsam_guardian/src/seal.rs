// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Keys at rest (design §6.1, §6.3). Two files only on a guardian machine:
//!
//! * the **machine key** (X25519), sealed by the operator's passphrase:
//!   `JGMK1 ‖ m_cost u32 ‖ t_cost u32 ‖ p u32 ‖ salt 16 ‖ nonce 12 ‖
//!   ChaCha20-Poly1305(argon2id(passphrase, salt), secret 32)`;
//! * the **share**, sealed for the machine's public key:
//!   `JGSB1 ‖ ephemeral public 32 ‖ nonce 12 ‖ ChaCha20-Poly1305(k, payload)`,
//!   `k = HKDF-SHA256(salt = ephemeral ‖ recipient, ikm = X25519(e, R),
//!   info = "jetsam-guardian/seal/v1")`, the associated data naming the
//!   share (generation, index).
//!
//! Écart with the design: the design names `age` (X25519 + scrypt); the
//! `age` crate is not in this workspace's offline registry, so the same
//! construction is written with the crates the node already links
//! (`x25519-dalek`, `hkdf`, `chacha20poly1305`, `argon2`).

use chacha20poly1305::aead::{Aead, KeyInit, Payload};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroizing;

const MACHINE_MAGIC: &[u8; 5] = b"JGMK1";
const SEAL_MAGIC: &[u8; 5] = b"JGSB1";
const SEAL_INFO: &[u8] = b"jetsam-guardian/seal/v1";
const MACHINE_AAD: &[u8] = b"jetsam-guardian/machine-key/v1";

fn random<const N: usize>() -> anyhow::Result<[u8; N]> {
    let mut out = [0u8; N];
    getrandom::getrandom(&mut out).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
    Ok(out)
}

fn public_of(secret: &[u8; 32]) -> [u8; 32] {
    x25519_dalek::x25519(*secret, x25519_dalek::X25519_BASEPOINT_BYTES)
}

/// Argon2id cost of a machine key: 256 MiB, 3 passes (≈ scrypt 2^18).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdfCost {
    pub m_kib: u32,
    pub t: u32,
    pub p: u32,
}

impl KdfCost {
    pub const PRODUCTION: Self = Self {
        m_kib: 256 * 1024,
        t: 3,
        p: 1,
    };
    /// Tests and the private rehearsal only.
    pub const TEST: Self = Self { m_kib: 64, t: 1, p: 1 };
}

/// A machine key: the X25519 secret (zeroized on drop) and its public key.
pub struct MachineKey {
    secret: Zeroizing<[u8; 32]>,
    pub public: [u8; 32],
}

impl MachineKey {
    /// A new key from the OS generator.
    pub fn generate() -> anyhow::Result<Self> {
        Ok(Self::from_secret(random::<32>()?))
    }

    /// From a secret (tests, TEST dealings).
    pub fn from_secret(secret: [u8; 32]) -> Self {
        let secret = Zeroizing::new(secret);
        let public = public_of(&secret);
        Self { secret, public }
    }

    /// The fingerprint printed for the operator (SHA-256 of the public key, 8 bytes).
    pub fn fingerprint(&self) -> String {
        use sha2::Digest;
        hex::encode(&sha2::Sha256::digest(self.public)[..8])
    }
}

/// Seal a machine key with a passphrase.
pub fn seal_machine_key(key: &MachineKey, passphrase: &[u8], cost: KdfCost) -> anyhow::Result<Vec<u8>> {
    let salt = random::<16>()?;
    let nonce = random::<12>()?;
    let k = derive(passphrase, &salt, cost)?;
    let ct = ChaCha20Poly1305::new(Key::from_slice(&k[..]))
        .encrypt(Nonce::from_slice(&nonce), Payload { msg: &key.secret[..], aad: MACHINE_AAD })
        .map_err(|_| anyhow::anyhow!("seal: encryption failed"))?;
    let mut out = MACHINE_MAGIC.to_vec();
    out.extend_from_slice(&cost.m_kib.to_be_bytes());
    out.extend_from_slice(&cost.t.to_be_bytes());
    out.extend_from_slice(&cost.p.to_be_bytes());
    out.extend_from_slice(&salt);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ct);
    Ok(out)
}

fn derive(passphrase: &[u8], salt: &[u8; 16], cost: KdfCost) -> anyhow::Result<Zeroizing<[u8; 32]>> {
    let params = argon2::Params::new(cost.m_kib, cost.t, cost.p, Some(32))
        .map_err(|e| anyhow::anyhow!("argon2 parameters: {e}"))?;
    let argon = argon2::Argon2::new(argon2::Algorithm::Argon2id, argon2::Version::V0x13, params);
    let mut out = Zeroizing::new([0u8; 32]);
    argon
        .hash_password_into(passphrase, salt, &mut out[..])
        .map_err(|e| anyhow::anyhow!("argon2: {e}"))?;
    Ok(out)
}

/// M-12 (external audit of the robots): the KDF cost a sealed machine key names (its header), before any derivation.
pub fn sealed_cost(blob: &[u8]) -> anyhow::Result<KdfCost> {
    anyhow::ensure!(blob.len() == 5 + 12 + 16 + 12 + 32 + 16 && &blob[..5] == MACHINE_MAGIC, "not a sealed machine key");
    let u = |i: usize| u32::from_be_bytes(blob[i..i + 4].try_into().expect("4"));
    Ok(KdfCost { m_kib: u(5), t: u(9), p: u(13) })
}

/// M-12: off the private rehearsal a machine key sealed below the production cost (`--test-kdf`, a passphrase an
/// offline search breaks in no time) is refused: the machine does not start with it.
pub fn check_machine_key_cost(blob: &[u8], profile: crate::Profile) -> anyhow::Result<()> {
    let c = sealed_cost(blob)?;
    let p = KdfCost::PRODUCTION;
    if profile.strict() && (c.m_kib < p.m_kib || c.t < p.t) {
        anyhow::bail!(
            "the machine key is sealed with a TEST cost (argon2id {} KiB, {} passes; production {} KiB, {} passes): refused on {profile}, \
             seal a new one without --test-kdf",
            c.m_kib,
            c.t,
            p.m_kib,
            p.t
        );
    }
    Ok(())
}

/// M-12: `bytes` written to a NEW file `path` (never over one, never through a symbolic link), created 0600 whatever
/// the umask, synced with its directory: a secret at rest is its owner's alone.
pub fn write_new_private(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|e| anyhow::anyhow!("{}: {e} (never written over a file already there)", path.display()))?;
    // the mode asked is masked by the umask at creation: set it again on the descriptor
    use std::os::unix::fs::PermissionsExt;
    f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::File::open(dir).and_then(|d| d.sync_all()).map_err(|e| anyhow::anyhow!("{}: sync: {e}", dir.display()))?;
    }
    Ok(())
}

/// Open a sealed machine key (a wrong passphrase fails).
pub fn open_machine_key(blob: &[u8], passphrase: &[u8]) -> anyhow::Result<MachineKey> {
    let cost = sealed_cost(blob)?;
    let salt: [u8; 16] = blob[17..33].try_into().expect("16");
    let nonce = &blob[33..45];
    let k = derive(passphrase, &salt, cost)?;
    let plain = Zeroizing::new(
        ChaCha20Poly1305::new(Key::from_slice(&k[..]))
            .decrypt(Nonce::from_slice(nonce), Payload { msg: &blob[45..], aad: MACHINE_AAD })
            .map_err(|_| anyhow::anyhow!("wrong passphrase or damaged machine key"))?,
    );
    let secret: [u8; 32] = plain[..].try_into().map_err(|_| anyhow::anyhow!("machine key length"))?;
    Ok(MachineKey::from_secret(secret))
}

/// Seal `plaintext` for `recipient` (X25519 public key) under `aad`.
pub fn seal_to(recipient: &[u8; 32], plaintext: &[u8], aad: &[u8]) -> anyhow::Result<Vec<u8>> {
    let ephemeral = Zeroizing::new(random::<32>()?);
    let e_pub = public_of(&ephemeral);
    let shared = Zeroizing::new(x25519_dalek::x25519(*ephemeral, *recipient));
    let k = seal_key(&shared, &e_pub, recipient)?;
    let nonce = random::<12>()?;
    let ct = ChaCha20Poly1305::new(Key::from_slice(&k[..]))
        .encrypt(Nonce::from_slice(&nonce), Payload { msg: plaintext, aad })
        .map_err(|_| anyhow::anyhow!("seal: encryption failed"))?;
    let mut out = SEAL_MAGIC.to_vec();
    out.extend_from_slice(&e_pub);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ct);
    Ok(out)
}

fn seal_key(shared: &[u8; 32], e_pub: &[u8; 32], recipient: &[u8; 32]) -> anyhow::Result<Zeroizing<[u8; 32]>> {
    anyhow::ensure!(shared.iter().any(|b| *b != 0), "degenerate shared secret");
    let mut salt = [0u8; 64];
    salt[..32].copy_from_slice(e_pub);
    salt[32..].copy_from_slice(recipient);
    let hk = Hkdf::<Sha256>::new(Some(&salt), shared);
    let mut k = Zeroizing::new([0u8; 32]);
    hk.expand(SEAL_INFO, &mut k[..]).map_err(|_| anyhow::anyhow!("hkdf"))?;
    Ok(k)
}

/// Open what was sealed for `key` under `aad`.
pub fn open_sealed(key: &MachineKey, blob: &[u8], aad: &[u8]) -> anyhow::Result<Zeroizing<Vec<u8>>> {
    anyhow::ensure!(blob.len() > 5 + 32 + 12 + 16 && &blob[..5] == SEAL_MAGIC, "not a sealed blob");
    let e_pub: [u8; 32] = blob[5..37].try_into().expect("32");
    let shared = Zeroizing::new(x25519_dalek::x25519(*key.secret, e_pub));
    let k = seal_key(&shared, &e_pub, &key.public)?;
    let plain = ChaCha20Poly1305::new(Key::from_slice(&k[..]))
        .decrypt(Nonce::from_slice(&blob[37..49]), Payload { msg: &blob[49..], aad })
        .map_err(|_| anyhow::anyhow!("this blob is not sealed for this machine key (or it is damaged)"))?;
    Ok(Zeroizing::new(plain))
}

/// The associated data of share `index` of generation `generation`.
pub fn share_aad(generation: u32, index: u32) -> Vec<u8> {
    format!("jetsam-guardian/share/g{generation}/i{index}").into_bytes()
}

/// A share's sealed payload: `index u32 BE ‖ generation u32 BE ‖ scalar 32`.
pub fn share_payload(generation: u32, index: u32, scalar: &[u8; 32]) -> Zeroizing<Vec<u8>> {
    let mut out = Zeroizing::new(Vec::with_capacity(40));
    out.extend_from_slice(&index.to_be_bytes());
    out.extend_from_slice(&generation.to_be_bytes());
    out.extend_from_slice(scalar);
    out
}

#[cfg(test)]
#[path = "seal_tests.rs"]
mod tests;
