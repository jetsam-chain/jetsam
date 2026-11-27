// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! 5-of-7 threshold BLS on `blst` (decision D12, prototype T0.5): BLS12-381
//! min-pk basic scheme (key in G1, 48 bytes; signature in G2, 96 bytes),
//! hash to G2 under `GUARDIAN_DST`; Shamir over Fr (polynomial of degree 4,
//! share i = f(i)), Feldman commitments `C_k = a_k · G1` (`C_0` = group key),
//! combination by Lagrange interpolation at 0 in the exponent.
//!
//! Differences with the T0.5 prototype: every secret scalar lives in a
//! [`zeroize::Zeroizing`] buffer or a `blst` `SecretKey` (zeroized on drop by
//! `blst` 0.3.16, `#[zeroize(drop)]`), and the temporary `blst_fr` of the
//! dealing and of the reconstruction are overwritten before they go out of
//! scope (the prototype did not).

use blst::min_pk::{PublicKey, SecretKey, Signature};
use blst::*;
use zeroize::Zeroizing;

use crate::GUARDIAN_DST;

pub const THRESHOLD: usize = 5;
pub const PARTIES: u32 = 7;

#[derive(Debug, PartialEq, Eq)]
pub enum ThresholdError {
    NotEnoughPartials { got: usize, need: usize },
    DuplicateIndex(u32),
    BadIndex(u32),
    WrongCoefficientCount(usize),
    ZeroShare(u32),
    BadPoint,
    BadScalar,
}

impl std::fmt::Display for ThresholdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ThresholdError {}

/// One share: its index 1..=7 and its secret scalar (big-endian, zeroized
/// on drop).
pub struct Share {
    pub index: u32,
    secret: Zeroizing<[u8; 32]>,
}

impl Share {
    /// A share from its big-endian scalar (refuses 0 and ≥ r).
    pub fn from_bytes(index: u32, secret: &[u8; 32]) -> Result<Self, ThresholdError> {
        if index == 0 || index > PARTIES {
            return Err(ThresholdError::BadIndex(index));
        }
        SecretKey::from_bytes(secret).map_err(|_| ThresholdError::BadScalar)?;
        Ok(Self { index, secret: Zeroizing::new(*secret) })
    }

    /// The scalar (zeroized on drop), for sealing.
    pub fn to_bytes(&self) -> Zeroizing<[u8; 32]> {
        Zeroizing::new(*self.secret)
    }

    fn secret_key(&self) -> SecretKey {
        SecretKey::from_bytes(&self.secret[..]).expect("checked at construction")
    }

    /// The share's public key, compressed.
    pub fn public_key(&self) -> [u8; 48] {
        self.secret_key().sk_to_pk().compress()
    }
}

/// A dealing: the group key, the Feldman commitments, the 7 shares.
pub struct Dealing {
    pub group_key: [u8; 48],
    /// `C_k`, compressed, k = 0..=4 (`C_0` = group key).
    pub commitments: Vec<[u8; 48]>,
    pub shares: Vec<Share>,
}

/// Deal the shares of the polynomial whose coefficients are `coeffs`
/// (big-endian scalars; `coeffs[0]` is the group secret).
pub fn deal(coeffs: &[Zeroizing<[u8; 32]>]) -> Result<Dealing, ThresholdError> {
    if coeffs.len() != THRESHOLD {
        return Err(ThresholdError::WrongCoefficientCount(coeffs.len()));
    }
    let mut keys = Vec::with_capacity(THRESHOLD);
    for c in coeffs {
        keys.push(SecretKey::from_bytes(&c[..]).map_err(|_| ThresholdError::BadScalar)?);
    }
    let mut a: Vec<blst_fr> = coeffs.iter().map(|c| fr_from_be(c)).collect();
    let mut shares = Vec::with_capacity(PARTIES as usize);
    for i in 1..=PARTIES {
        // Horner: f(i) = (((a4 i + a3) i + a2) i + a1) i + a0
        let x = fr_from_u64(u64::from(i));
        let mut acc = a[THRESHOLD - 1];
        for k in (0..THRESHOLD - 1).rev() {
            let mut t = blst_fr::default();
            unsafe {
                blst_fr_mul(&mut t, &acc, &x);
                blst_fr_add(&mut acc, &t, &a[k]);
            }
            wipe(&mut t);
        }
        let be = be_from_fr(&acc);
        wipe(&mut acc);
        let share = Share::from_bytes(i, &be).map_err(|_| ThresholdError::ZeroShare(i))?;
        shares.push(share);
    }
    for x in &mut a {
        wipe(x);
    }
    let commitments = keys.iter().map(|k| k.sk_to_pk().compress()).collect::<Vec<_>>();
    Ok(Dealing { group_key: commitments[0], commitments, shares })
}

// ------------------------------------------------------------------ Fr helpers

fn wipe(x: &mut blst_fr) {
    unsafe { std::ptr::write_volatile(x, blst_fr::default()) };
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

fn fr_from_u64(x: u64) -> blst_fr {
    let mut r = blst_fr::default();
    let limbs = [x, 0u64, 0, 0];
    unsafe { blst_fr_from_uint64(&mut r, limbs.as_ptr()) };
    r
}

fn fr_from_be(be: &[u8; 32]) -> blst_fr {
    let mut s = blst_scalar::default(); // zeroized on drop by blst
    let mut r = blst_fr::default();
    unsafe {
        blst_scalar_from_bendian(&mut s, be.as_ptr());
        blst_fr_from_scalar(&mut r, &s);
    }
    r
}

fn be_from_fr(fr: &blst_fr) -> Zeroizing<[u8; 32]> {
    let mut s = blst_scalar::default();
    let mut be = Zeroizing::new([0u8; 32]);
    unsafe {
        blst_scalar_from_fr(&mut s, fr);
        blst_bendian_from_scalar(be.as_mut_ptr(), &s);
    }
    be
}

fn le_from_fr(fr: &blst_fr) -> Zeroizing<[u8; 32]> {
    let mut s = blst_scalar::default();
    let mut le = Zeroizing::new([0u8; 32]);
    unsafe {
        blst_scalar_from_fr(&mut s, fr);
        blst_lendian_from_scalar(le.as_mut_ptr(), &s);
    }
    le
}

/// Lagrange coefficients at 0 of the distinct non-zero `xs`.
fn lagrange_at_zero(xs: &[u32]) -> Result<Vec<blst_fr>, ThresholdError> {
    let mut seen = std::collections::BTreeSet::new();
    for x in xs {
        if *x == 0 || *x > PARTIES {
            return Err(ThresholdError::BadIndex(*x));
        }
        if !seen.insert(*x) {
            return Err(ThresholdError::DuplicateIndex(*x));
        }
    }
    let f: Vec<blst_fr> = xs.iter().map(|x| fr_from_u64(u64::from(*x))).collect();
    let mut out = Vec::with_capacity(xs.len());
    for m in 0..xs.len() {
        let mut num = fr_from_u64(1);
        let mut den = fr_from_u64(1);
        for (j, xj) in f.iter().enumerate() {
            if j == m {
                continue;
            }
            let mut d = blst_fr::default();
            let mut t = blst_fr::default();
            unsafe {
                blst_fr_mul(&mut t, &num, xj);
                num = t;
                blst_fr_sub(&mut d, xj, &f[m]);
                blst_fr_mul(&mut t, &den, &d);
                den = t;
            }
        }
        let mut inv = blst_fr::default();
        let mut lambda = blst_fr::default();
        unsafe {
            blst_fr_inverse(&mut inv, &den);
            blst_fr_mul(&mut lambda, &num, &inv);
        }
        out.push(lambda);
    }
    Ok(out)
}

fn p1_of(pk: &PublicKey) -> blst_p1 {
    let aff: blst_p1_affine = (*pk).into();
    let mut p = blst_p1::default();
    unsafe { blst_p1_from_affine(&mut p, &aff) };
    p
}

fn pk_of(p: &blst_p1) -> PublicKey {
    let mut aff = blst_p1_affine::default();
    unsafe { blst_p1_to_affine(&mut aff, p) };
    PublicKey::from(aff)
}

fn p2_of(sig: &Signature) -> blst_p2 {
    let aff: blst_p2_affine = (*sig).into();
    let mut p = blst_p2::default();
    unsafe { blst_p2_from_affine(&mut p, &aff) };
    p
}

fn sig_of(p: &blst_p2) -> Signature {
    let mut aff = blst_p2_affine::default();
    unsafe { blst_p2_to_affine(&mut aff, p) };
    Signature::from(aff)
}

/// The public key of share `index` from the commitments: `Σ C_k · index^k`.
pub fn share_public_key(index: u32, commitments: &[[u8; 48]]) -> Result<[u8; 48], ThresholdError> {
    if index == 0 || index > PARTIES {
        return Err(ThresholdError::BadIndex(index));
    }
    if commitments.len() != THRESHOLD {
        return Err(ThresholdError::WrongCoefficientCount(commitments.len()));
    }
    let pts: Vec<blst_p1> = commitments
        .iter()
        .map(|c| PublicKey::key_validate(c).map(|pk| p1_of(&pk)).map_err(|_| ThresholdError::BadPoint))
        .collect::<Result<_, _>>()?;
    let ile = u64::from(index).to_le_bytes();
    let mut acc = pts[pts.len() - 1];
    for k in (0..pts.len() - 1).rev() {
        let mut t = blst_p1::default();
        unsafe {
            blst_p1_mult(&mut t, &acc, ile.as_ptr(), 64);
            blst_p1_add_or_double(&mut acc, &t, &pts[k]);
        }
    }
    Ok(pk_of(&acc).compress())
}

/// Feldman check of one share against the commitments.
pub fn verify_share(share: &Share, commitments: &[[u8; 48]]) -> bool {
    share_public_key(share.index, commitments).is_ok_and(|pk| pk == share.public_key())
}

/// The partial signature of `msg` by `share`.
pub fn partial_sign(share: &Share, msg: &[u8]) -> [u8; 96] {
    share.secret_key().sign(msg, GUARDIAN_DST, &[]).compress()
}

/// The node's verification (T1.1 §4.3): signature group check, key
/// validation, `GUARDIAN_DST`, no augmentation; infinity refused.
pub fn verify(sig: &[u8; 96], msg: &[u8], key: &[u8; 48]) -> bool {
    let Ok(pk) = PublicKey::key_validate(key) else {
        return false;
    };
    let Ok(sig) = Signature::from_bytes(sig) else {
        return false;
    };
    sig.verify(true, msg, GUARDIAN_DST, &[], &pk, true) == BLST_ERROR::BLST_SUCCESS
}

/// A partial checked against the share key derived from the commitments.
pub fn verify_partial(index: u32, sig: &[u8; 96], msg: &[u8], commitments: &[[u8; 48]]) -> bool {
    share_public_key(index, commitments).is_ok_and(|pk| verify(sig, msg, &pk))
}

/// Lagrange interpolation at 0 over all the given partials (no count check).
pub fn interpolate(partials: &[(u32, [u8; 96])]) -> Result<[u8; 96], ThresholdError> {
    let xs: Vec<u32> = partials.iter().map(|(i, _)| *i).collect();
    let lambdas = lagrange_at_zero(&xs)?;
    let mut acc = blst_p2::default();
    for (m, (_, bytes)) in partials.iter().enumerate() {
        let sig = Signature::from_bytes(bytes).map_err(|_| ThresholdError::BadPoint)?;
        let le = le_from_fr(&lambdas[m]);
        let mut term = blst_p2::default();
        unsafe { blst_p2_mult(&mut term, &p2_of(&sig), le.as_ptr(), 255) };
        if m == 0 {
            acc = term;
        } else {
            let mut t = blst_p2::default();
            unsafe { blst_p2_add_or_double(&mut t, &acc, &term) };
            acc = t;
        }
    }
    Ok(sig_of(&acc).compress())
}

/// At least 5 partials (distinct indices 1..=7) into the group signature.
pub fn combine(partials: &[(u32, [u8; 96])]) -> Result<[u8; 96], ThresholdError> {
    if partials.len() < THRESHOLD {
        return Err(ThresholdError::NotEnoughPartials {
            got: partials.len(),
            need: THRESHOLD,
        });
    }
    interpolate(partials)
}

/// The group secret from at least 5 shares (Lagrange on the scalars): only
/// for the re-sharing ceremony, offline.
pub fn reconstruct_secret(shares: &[&Share]) -> Result<Zeroizing<[u8; 32]>, ThresholdError> {
    if shares.len() < THRESHOLD {
        return Err(ThresholdError::NotEnoughPartials { got: shares.len(), need: THRESHOLD });
    }
    let xs: Vec<u32> = shares.iter().map(|s| s.index).collect();
    let mut lambdas = lagrange_at_zero(&xs)?;
    let mut acc = blst_fr::default();
    for (m, share) in shares.iter().enumerate() {
        let mut y = fr_from_be(&share.secret);
        let mut t = blst_fr::default();
        let mut sum = blst_fr::default();
        unsafe {
            blst_fr_mul(&mut t, &lambdas[m], &y);
            blst_fr_add(&mut sum, &acc, &t);
        }
        acc = sum;
        wipe(&mut y);
        wipe(&mut t);
        wipe(&mut sum);
    }
    for l in &mut lambdas {
        wipe(l);
    }
    let out = be_from_fr(&acc);
    wipe(&mut acc);
    Ok(out)
}

/// EIP-2537 G1 (128 bytes) of a compressed key.
pub fn g1_eip2537(key: &[u8; 48]) -> Result<[u8; 128], ThresholdError> {
    let pk = PublicKey::key_validate(key).map_err(|_| ThresholdError::BadPoint)?;
    let ser = pk.serialize(); // x ‖ y, 48 bytes each, big-endian
    let mut o = [0u8; 128];
    o[16..64].copy_from_slice(&ser[..48]);
    o[64 + 16..128].copy_from_slice(&ser[48..]);
    Ok(o)
}

/// EIP-2537 G2 (256 bytes) of a compressed signature: coordinates in the
/// order `x.c0 ‖ x.c1 ‖ y.c0 ‖ y.c1` (blst serializes c1 first).
pub fn g2_eip2537(sig: &[u8; 96]) -> Result<[u8; 256], ThresholdError> {
    let sig = Signature::from_bytes(sig).map_err(|_| ThresholdError::BadPoint)?;
    let ser = sig.serialize(); // x.c1 ‖ x.c0 ‖ y.c1 ‖ y.c0
    let mut o = [0u8; 256];
    o[16..64].copy_from_slice(&ser[48..96]);
    o[64 + 16..128].copy_from_slice(&ser[0..48]);
    o[128 + 16..192].copy_from_slice(&ser[144..192]);
    o[192 + 16..256].copy_from_slice(&ser[96..144]);
    Ok(o)
}

/// The direct signature by a group secret (ceremony checks, TEST dealings).
pub fn sign_with_secret(secret: &[u8; 32], msg: &[u8]) -> Result<[u8; 96], ThresholdError> {
    let sk = SecretKey::from_bytes(secret).map_err(|_| ThresholdError::BadScalar)?;
    Ok(sk.sign(msg, GUARDIAN_DST, &[]).compress())
}

/// A uniformly random scalar from 64 bytes of key material (`KeyGen`, HKDF).
pub fn scalar_from_ikm(ikm: &[u8], info: &[u8]) -> Result<Zeroizing<[u8; 32]>, ThresholdError> {
    let sk = SecretKey::key_gen(ikm, info).map_err(|_| ThresholdError::BadScalar)?;
    Ok(Zeroizing::new(sk.to_bytes()))
}

/// The T0.5 TEST coefficients of `label` (fixed IKM, never a real key).
pub fn test_coefficients(label: &str) -> Vec<Zeroizing<[u8; 32]>> {
    (0..THRESHOLD)
        .map(|k| {
            let ikm = format!("jetsam-v17-T0.5-TEST-ONLY/{label}/coefficient-{k}/padding-to-32-bytes");
            scalar_from_ikm(ikm.as_bytes(), &[]).expect("ikm >= 32 bytes")
        })
        .collect()
}

/// The TEST dealing of `label`; `"fixture"` is the rehearsal's group key
/// (`REHEARSAL_GUARDIAN_GROUP_KEY` of the node, T0.5).
pub fn test_dealing(label: &str) -> Dealing {
    deal(&test_coefficients(label)).expect("a test dealing")
}

#[cfg(test)]
#[path = "threshold_tests.rs"]
mod tests;
