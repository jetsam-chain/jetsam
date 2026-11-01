// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The key ceremony (design §6.2, T5.4): offline, in RAM only.
//!
//! * entropy: 64 bytes of the OS generator mixed with the SHA-512 of at least
//!   100 dice throws typed by the operator; the 5 coefficients derived from it
//!   by `KeyGen` (HKDF), one information string per coefficient;
//! * Shamir 5 of 7, Feldman commitments, internal checks (each share against
//!   the commitments; the 21 subsets of 5 give the direct signature of the
//!   test message; 4 partials fail);
//! * the test message is OUTSIDE the statement domain (it is not the hash of
//!   416 bytes), so its signature can never enter an equivocation;
//! * each share sealed for its machine key; only public files and sealed
//!   blobs are written;
//! * the revocation certificate (D34): two contradictory statements at height
//!   2^63, rule E1, signed by the group secret at the ceremony;
//! * re-sharing at a constant group key (rotation): the secret rebuilt from 5
//!   shares, a new polynomial of the same constant term.

use std::path::Path;

use sha2::{Digest, Sha512};
use zeroize::Zeroizing;

use crate::threshold::{self, Dealing, Share, THRESHOLD};
use crate::{Domain, GuardianStatement, GUARDIAN_STATEMENT_DOMAIN, GUARDIAN_STATEMENT_VERSION};
use jetsam_chain::consensus::guardian::{u256_word, GuardianEquivocation};

/// At least this many dice throws (digits 1 to 6).
pub const MIN_DICE: usize = 100;

/// `IKM = SHA-512("jetsam-guardian/ceremony/ikm/v1" ‖ os ‖ SHA-512(dice))`.
pub fn ikm_from_entropy(os: &[u8; 64], dice: &str) -> anyhow::Result<Zeroizing<[u8; 64]>> {
    let throws: Vec<char> = dice.chars().filter(|c| !c.is_whitespace()).collect();
    anyhow::ensure!(throws.len() >= MIN_DICE, "{} dice throws, at least {MIN_DICE} are needed", throws.len());
    anyhow::ensure!(throws.iter().all(|c| ('1'..='6').contains(c)), "dice throws are digits 1 to 6");
    let dice_digest = Sha512::digest(throws.iter().collect::<String>().as_bytes());
    let mut h = Sha512::new();
    h.update(b"jetsam-guardian/ceremony/ikm/v1");
    h.update(os);
    h.update(dice_digest);
    let mut out = Zeroizing::new([0u8; 64]);
    out.copy_from_slice(&h.finalize());
    Ok(out)
}

/// The five coefficients of a dealing from the IKM.
pub fn coefficients(ikm: &[u8; 64]) -> anyhow::Result<Vec<Zeroizing<[u8; 32]>>> {
    (0..THRESHOLD)
        .map(|k| {
            let info = format!("jetsam-guardian/ceremony/coefficient-{k}");
            threshold::scalar_from_ikm(ikm, info.as_bytes()).map_err(|e| anyhow::anyhow!("KeyGen: {e}"))
        })
        .collect()
}

/// `m_test = keccak256("jetsam-guardian/ceremony-test/v1" ‖ C_0 ‖ date)`.
pub fn test_message(group_key: &[u8; 48], date: &str) -> [u8; 32] {
    let mut pre = b"jetsam-guardian/ceremony-test/v1".to_vec();
    pre.extend_from_slice(group_key);
    pre.extend_from_slice(date.as_bytes());
    crate::keccak256(&pre)
}

/// The internal checks of a fresh dealing; returns the test signature.
pub fn check_dealing(dealing: &Dealing, secret: &[u8; 32], m_test: &[u8; 32]) -> anyhow::Result<[u8; 96]> {
    for s in &dealing.shares {
        anyhow::ensure!(threshold::verify_share(s, &dealing.commitments), "share {} fails its commitments", s.index);
    }
    let direct = threshold::sign_with_secret(secret, m_test)?;
    anyhow::ensure!(threshold::verify(&direct, m_test, &dealing.group_key), "the direct test signature fails");
    let partials: Vec<(u32, [u8; 96])> = dealing.shares.iter().map(|s| (s.index, threshold::partial_sign(s, m_test))).collect();
    let n = partials.len();
    let mut subsets = 0;
    for mask in 0u32..(1 << n) {
        if mask.count_ones() as usize != THRESHOLD {
            continue;
        }
        let chosen: Vec<_> = (0..n).filter(|i| mask & (1 << i) != 0).map(|i| partials[i]).collect();
        anyhow::ensure!(threshold::combine(&chosen)? == direct, "a 5-subset does not give the direct signature");
        subsets += 1;
        let four = &chosen[..THRESHOLD - 1];
        anyhow::ensure!(!threshold::verify(&threshold::interpolate(four)?, m_test, &dealing.group_key), "4 partials signed");
    }
    anyhow::ensure!(subsets == 21, "21 subsets of 5 among 7");
    Ok(direct)
}

/// The revocation certificate (D34): the 0x21 body (rule E1, 1 025 bytes)
/// and the L1 hints (the two signatures in EIP-2537 form, 512 bytes).
pub struct RevocationCertificate {
    pub body: Vec<u8>,
    pub hints: Vec<u8>,
}

/// Build the certificate with the group secret (at the ceremony only).
pub fn revocation_certificate(secret: &[u8; 32], domain: &Domain) -> anyhow::Result<RevocationCertificate> {
    let statement = |n: u8| GuardianStatement {
        domain: GUARDIAN_STATEMENT_DOMAIN,
        network: domain.network,
        l1_chain_id: u256_word(domain.l1_chain_id),
        l1_bridge: domain.l1_bridge,
        schema_version: GUARDIAN_STATEMENT_VERSION,
        height: 1u64 << 63,
        block_id: crate::keccak256(format!("jetsam-guardian/revocation/{n}").as_bytes()),
        cum_work: [0u8; 32],
        headers_digest: [0u8; 32],
        anchor_root: [0u8; 32],
        deposit_root: [0u8; 32],
        deposit_count: 0,
        not_after: 0,
    }
    .encode();
    let (mut s1, mut s2) = (statement(1), statement(2));
    if s1 > s2 {
        std::mem::swap(&mut s1, &mut s2);
    }
    let sig1 = threshold::sign_with_secret(secret, &crate::keccak256(&s1))?;
    let sig2 = threshold::sign_with_secret(secret, &crate::keccak256(&s2))?;
    let eq = GuardianEquivocation { s1, sig1, s2, sig2, witnesses: None };
    let mut hints = threshold::g2_eip2537(&sig1)?.to_vec();
    hints.extend_from_slice(&threshold::g2_eip2537(&sig2)?);
    Ok(RevocationCertificate { body: eq.encode(), hints })
}

/// Re-share at a constant group key: the secret from `old` (≥ 5 shares), a
/// new polynomial with the same constant term and new coefficients `rest`
/// (4 scalars).
pub fn reshare(old: &[&Share], rest: &[Zeroizing<[u8; 32]>]) -> anyhow::Result<Dealing> {
    anyhow::ensure!(rest.len() == THRESHOLD - 1, "a re-sharing takes {} new coefficients", THRESHOLD - 1);
    let secret = threshold::reconstruct_secret(old).map_err(|e| anyhow::anyhow!("reconstruction: {e}"))?;
    let mut coeffs = vec![secret];
    coeffs.extend(rest.iter().cloned());
    threshold::deal(&coeffs).map_err(|e| anyhow::anyhow!("dealing: {e}"))
}

/// What a ceremony writes: for each machine i, `share-g<gen>-i<i>.sealed`
/// (sealed for its machine key), and `public.json` (group key, commitments,
/// the share keys, the EIP-2537 key, the sealed files' SHA-256). Nothing else.
pub fn write_outputs(dir: &Path, dealing: &Dealing, machine_keys: &[[u8; 32]], generation: u32) -> anyhow::Result<serde_json::Value> {
    use sha2::Sha256;
    anyhow::ensure!(machine_keys.len() == dealing.shares.len(), "one machine key per share");
    std::fs::create_dir_all(dir)?;
    let mut sealed = serde_json::Map::new();
    let mut share_keys = Vec::new();
    for (share, mk) in dealing.shares.iter().zip(machine_keys) {
        let payload = crate::seal::share_payload(generation, share.index, &share.to_bytes());
        let blob = crate::seal::seal_to(mk, &payload, &crate::seal::share_aad(generation, share.index))?;
        let name = format!("share-g{generation}-i{}.sealed", share.index);
        std::fs::write(dir.join(&name), &blob)?;
        sealed.insert(name, serde_json::json!(hex::encode(Sha256::digest(&blob))));
        share_keys.push(hex::encode(share.public_key()));
    }
    let public = serde_json::json!({
        "generation": generation,
        "group_key": hex::encode(dealing.group_key),
        "group_key_eip2537": hex::encode(threshold::g1_eip2537(&dealing.group_key)?),
        "commitments": dealing.commitments.iter().map(hex::encode).collect::<Vec<_>>(),
        "share_keys": share_keys,
        "machine_keys": machine_keys.iter().map(hex::encode).collect::<Vec<_>>(),
        "sealed_sha256": sealed,
    });
    std::fs::write(dir.join("public.json"), serde_json::to_vec_pretty(&public)?)?;
    Ok(public)
}

#[cfg(test)]
#[path = "ceremony_tests.rs"]
mod tests;
