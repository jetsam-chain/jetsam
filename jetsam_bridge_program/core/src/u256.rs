// SPDX-License-Identifier: Apache-2.0
//! 256-bit little-endian arithmetic, as `jetsam_chain::consensus::difficulty`.

pub fn le256_lt(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut i = 32;
    while i > 0 {
        i -= 1;
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    false
}

fn is_zero(a: &[u8; 32]) -> bool {
    a.iter().all(|b| *b == 0)
}

fn shl1(a: &mut [u8; 32]) {
    let mut c = 0u8;
    for b in a.iter_mut() {
        let n = *b >> 7;
        *b = (*b << 1) | c;
        c = n;
    }
}

fn sub_assign(a: &mut [u8; 32], b: &[u8; 32]) {
    let mut borrow = 0i16;
    for i in 0..32 {
        let d = a[i] as i16 - b[i] as i16 - borrow;
        if d < 0 {
            a[i] = (d + 256) as u8;
            borrow = 1;
        } else {
            a[i] = d as u8;
            borrow = 0;
        }
    }
}

fn div(n: &[u8; 32], d: &[u8; 32]) -> [u8; 32] {
    let mut q = [0u8; 32];
    let mut r = [0u8; 32];
    for bit in (0..256).rev() {
        shl1(&mut r);
        if (n[bit / 8] >> (bit % 8)) & 1 == 1 {
            r[0] |= 1;
        }
        if !le256_lt(&r, d) {
            sub_assign(&mut r, d);
            q[bit / 8] |= 1 << (bit % 8);
        }
    }
    q
}

fn add_one_saturating(a: &[u8; 32]) -> [u8; 32] {
    let mut r = *a;
    for b in r.iter_mut() {
        let (n, c) = b.overflowing_add(1);
        *b = n;
        if !c {
            return r;
        }
    }
    [0xFF; 32]
}

/// `block_work(target) = floor((2^256-1)/target) + 1`, saturating.
pub fn block_work(target: &[u8; 32]) -> [u8; 32] {
    if is_zero(target) {
        return [0u8; 32];
    }
    add_one_saturating(&div(&[0xFF; 32], target))
}

/// `add_work`, saturating.
pub fn add(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut r = [0u8; 32];
    let mut c = 0u16;
    for i in 0..32 {
        let s = a[i] as u16 + b[i] as u16 + c;
        r[i] = s as u8;
        c = s >> 8;
    }
    if c != 0 { [0xFF; 32] } else { r }
}

/// a - b, panics on underflow (fork-base work removal must never underflow).
pub fn sub(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    assert!(!le256_lt(a, b), "work underflow");
    let mut r = *a;
    sub_assign(&mut r, b);
    r
}

/// Little-endian u256 -> 32-byte big-endian ABI word.
pub fn to_be(a: &[u8; 32]) -> [u8; 32] {
    let mut r = [0u8; 32];
    for i in 0..32 {
        r[i] = a[31 - i];
    }
    r
}

/// 32-byte big-endian ABI word -> little-endian u256.
pub fn from_be(a: &[u8; 32]) -> [u8; 32] {
    to_be(a)
}

/// `a ≥ k × b`, exact (the product on 320 bits).
pub fn ge_mul(a: &[u8; 32], k: u64, b: &[u8; 32]) -> bool {
    let limb = |x: &[u8; 32], i: usize| u64::from_le_bytes(x[8 * i..8 * i + 8].try_into().unwrap());
    let mut product = [0u64; 5];
    let mut carry = 0u128;
    for (i, p) in product.iter_mut().take(4).enumerate() {
        let t = limb(b, i) as u128 * k as u128 + carry;
        *p = t as u64;
        carry = t >> 64;
    }
    product[4] = carry as u64;
    if product[4] != 0 {
        return false;
    }
    for i in (0..4).rev() {
        let x = limb(a, i);
        if x != product[i] {
            return x > product[i];
        }
    }
    true
}
