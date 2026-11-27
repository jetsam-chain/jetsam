// SPDX-License-Identifier: Apache-2.0
//! Software GF(2^128) multiplication, flat (GCM) basis.
//!
//! Field: GF(2)[x] / (x^128 + x^7 + x^2 + x + 1). Bit `i` of the `u128` is the
//! coefficient of x^i. This is the representation of
//! `jetsam_core::hardware::clmul_gcm` (used by the Poseidon2b permutation and so
//! by TowerWalk) and of `jetsam_ivc_core::field::gf2_128::F128`
//! (`lo` = bits 0..63, `hi` = bits 64..127).
//!
//! Variants (no hardware carry-less multiply anywhere):
//! - [`Repo`]   : the repository's own soft fallback, copied verbatim
//!                (`soft_clmul::clmul_block128` + `reduce_gcm_256`, u128 code).
//! - [`Naive`]  : bit-serial shift-and-XOR, 128 steps, 32-bit limbs.
//! - [`Comb4`]  : left-to-right comb with a 4-bit window (16-entry table built
//!                per multiplication), 32-bit limbs.
//! - [`CtKara`] : 32x32 carry-less products emulated with integer MUL on
//!                bit-masked operands (holes every 4 bits, BearSSL "ctmul"
//!                technique), two levels of Karatsuba (9 such products).

pub trait Gf {
    const NAME: &'static str;
    fn mul(a: u128, b: u128) -> u128;
    fn sq(a: u128) -> u128;
}

// ---------------------------------------------------------------------------
// Repository soft path, verbatim from jetsam_core/src/hardware.rs (v1.6.0).
// ---------------------------------------------------------------------------
pub mod repo {
    /// Carry-less multiply by the constant 0x87 (x^7 + x^2 + x + 1).
    #[inline(always)]
    const fn clmul_u128_by_87(a: u128) -> u128 {
        a ^ (a << 1) ^ (a << 2) ^ (a << 7)
    }

    #[inline(always)]
    pub const fn reduce_gcm_256(x_hi: u128, x_lo: u128) -> u128 {
        let x1_lo = x_hi as u64 as u128;
        let x1_hi = (x_hi >> 64) as u64 as u128;
        let v1 = clmul_u128_by_87(x1_lo);
        let v2 = clmul_u128_by_87(x1_hi);
        let v2_shift64 = v2 << 64;
        let v2_overflow = v2 >> 64;
        let lo = x_lo ^ v1 ^ v2_shift64;
        let hi = v2_overflow;
        let v3 = clmul_u128_by_87(hi);
        lo ^ v3
    }

    #[inline(always)]
    pub fn square_flat_u128(a: u128) -> u128 {
        let lo = a as u64;
        let hi = (a >> 64) as u64;
        let s_lo = expand_bits_u64_to_u128(lo);
        let s_hi = expand_bits_u64_to_u128(hi);
        reduce_gcm_256(s_hi, s_lo)
    }

    #[inline(always)]
    const fn expand_bits_u64_to_u128(v: u64) -> u128 {
        let mut x = v as u128;
        x = (x | (x << 32)) & 0x00000000FFFFFFFF00000000FFFFFFFF;
        x = (x | (x << 16)) & 0x0000FFFF0000FFFF0000FFFF0000FFFF;
        x = (x | (x << 8)) & 0x00FF00FF00FF00FF00FF00FF00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F0F;
        x = (x | (x << 2)) & 0x33333333333333333333333333333333;
        x = (x | (x << 1)) & 0x55555555555555555555555555555555;
        x
    }

    #[inline(always)]
    pub fn clmul_64x64(a: u64, b: u64) -> u128 {
        let mut res = 0u128;
        let mut b = b as u128;
        let a = a as u128;
        while b != 0 {
            let lsb = b & (!b + 1);
            res ^= a * lsb;
            b ^= lsb;
        }
        res
    }

    #[inline(always)]
    pub fn clmul_block128(a: u128, b: u128) -> u128 {
        let a_lo = a as u64;
        let a_hi = (a >> 64) as u64;
        let b_lo = b as u64;
        let b_hi = (b >> 64) as u64;

        let v0 = clmul_64x64(a_lo, b_lo);
        let v1 = clmul_64x64(a_hi, b_hi);
        let mid = clmul_64x64(a_lo ^ a_hi, b_lo ^ b_hi) ^ v0 ^ v1;

        let x_lo = v0 ^ (mid << 64);
        let x_hi = v1 ^ (mid >> 64);

        reduce_gcm_256(x_hi, x_lo)
    }
}

// ---------------------------------------------------------------------------
// 32-bit limb helpers shared by the rewritten variants.
// ---------------------------------------------------------------------------
#[inline(always)]
fn to_w(a: u128) -> [u32; 4] {
    [a as u32, (a >> 32) as u32, (a >> 64) as u32, (a >> 96) as u32]
}

#[inline(always)]
fn from_w(w: [u32; 4]) -> u128 {
    (w[0] as u128) | ((w[1] as u128) << 32) | ((w[2] as u128) << 64) | ((w[3] as u128) << 96)
}

/// Reduce a 256-bit product (eight 32-bit limbs, little-endian) modulo
/// x^128 + x^7 + x^2 + x + 1. Same algebra as `ghash_reduce` in
/// jetsam_ivc_core, written on 32-bit limbs.
#[inline(always)]
fn reduce8(r: [u32; 8]) -> u128 {
    let h = [r[4], r[5], r[6], r[7]];
    let t0 = h[0] ^ (h[0] << 1) ^ (h[0] << 2) ^ (h[0] << 7);
    let t1 = h[1] ^ (h[1] << 1) ^ (h[0] >> 31) ^ (h[1] << 2) ^ (h[0] >> 30) ^ (h[1] << 7) ^ (h[0] >> 25);
    let t2 = h[2] ^ (h[2] << 1) ^ (h[1] >> 31) ^ (h[2] << 2) ^ (h[1] >> 30) ^ (h[2] << 7) ^ (h[1] >> 25);
    let t3 = h[3] ^ (h[3] << 1) ^ (h[2] >> 31) ^ (h[3] << 2) ^ (h[2] >> 30) ^ (h[3] << 7) ^ (h[2] >> 25);
    // Bits of h that the three shifts pushed past x^127 (at most 7 bits).
    let ov = (h[3] >> 31) ^ (h[3] >> 30) ^ (h[3] >> 25);
    let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    from_w([r[0] ^ t0 ^ corr, r[1] ^ t1, r[2] ^ t2, r[3] ^ t3])
}

/// Squaring is GF(2)-linear: spread every bit to an even position (byte table),
/// then reduce. Shared by the three rewritten variants.
const fn build_spread8() -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        let mut v = 0u16;
        let mut b = 0;
        while b < 8 {
            if (i >> b) & 1 == 1 {
                v |= 1 << (2 * b);
            }
            b += 1;
        }
        t[i] = v;
        i += 1;
    }
    t
}
static SPREAD8: [u16; 256] = build_spread8();

#[inline(always)]
pub fn sq_spread(a: u128) -> u128 {
    let w = to_w(a);
    let mut r = [0u32; 8];
    for i in 0..4 {
        let x = w[i];
        r[2 * i] = (SPREAD8[(x & 0xff) as usize] as u32)
            | ((SPREAD8[((x >> 8) & 0xff) as usize] as u32) << 16);
        r[2 * i + 1] = (SPREAD8[((x >> 16) & 0xff) as usize] as u32)
            | ((SPREAD8[(x >> 24) as usize] as u32) << 16);
    }
    reduce8(r)
}

// ---------------------------------------------------------------------------
// Variant "naive": bit-serial shift-and-XOR.
// ---------------------------------------------------------------------------
#[inline(never)]
pub fn mul_naive(a: u128, b: u128) -> u128 {
    let mut v = to_w(a);
    let bw = to_w(b);
    let mut z = [0u32; 4];
    for w in 0..4 {
        let mut bb = bw[w];
        for _ in 0..32 {
            let m = 0u32.wrapping_sub(bb & 1);
            z[0] ^= v[0] & m;
            z[1] ^= v[1] & m;
            z[2] ^= v[2] & m;
            z[3] ^= v[3] & m;
            // v <- v * x mod P
            let c = 0u32.wrapping_sub(v[3] >> 31);
            v[3] = (v[3] << 1) | (v[2] >> 31);
            v[2] = (v[2] << 1) | (v[1] >> 31);
            v[1] = (v[1] << 1) | (v[0] >> 31);
            v[0] = (v[0] << 1) ^ (0x87 & c);
            bb >>= 1;
        }
    }
    from_w(z)
}

// ---------------------------------------------------------------------------
// Variant "comb4": left-to-right comb, 4-bit window, per-call table.
// ---------------------------------------------------------------------------
#[inline(never)]
pub fn mul_comb4(a: u128, b: u128) -> u128 {
    let aw = to_w(a);
    let bw = to_w(b);
    // t[u] = u(x) * a(x), deg <= 130 -> five 32-bit limbs.
    let mut t = [[0u32; 5]; 16];
    t[1] = [aw[0], aw[1], aw[2], aw[3], 0];
    let mut u = 2;
    while u < 16 {
        let p = t[u >> 1];
        let q = [
            p[0] << 1,
            (p[1] << 1) | (p[0] >> 31),
            (p[2] << 1) | (p[1] >> 31),
            (p[3] << 1) | (p[2] >> 31),
            (p[4] << 1) | (p[3] >> 31),
        ];
        t[u] = q;
        t[u + 1] = [q[0] ^ aw[0], q[1] ^ aw[1], q[2] ^ aw[2], q[3] ^ aw[3], q[4]];
        u += 2;
    }
    let mut c = [0u32; 8];
    let mut k = 8u32;
    while k > 0 {
        k -= 1;
        let sh = 4 * k;
        for j in 0..4 {
            let e = &t[((bw[j] >> sh) & 0xF) as usize];
            c[j] ^= e[0];
            c[j + 1] ^= e[1];
            c[j + 2] ^= e[2];
            c[j + 3] ^= e[3];
            c[j + 4] ^= e[4];
        }
        if k != 0 {
            for i in (1..8).rev() {
                c[i] = (c[i] << 4) | (c[i - 1] >> 28);
            }
            c[0] <<= 4;
        }
    }
    reduce8(c)
}

// ---------------------------------------------------------------------------
// Variant "ctkara": integer-multiply carry-less 32x32 + 2-level Karatsuba.
// ---------------------------------------------------------------------------

/// 32x32 -> 64 carry-less product using 16 integer multiplications on operands
/// masked to every 4th bit. Each output position sums at most 8 one-bit
/// products, so the count fits in the 4-bit hole and never carries into the
/// next live position; its parity is the carry-less bit.
///
/// `inline(never)` is deliberate: inlined nine times into one basic block,
/// LLVM hoists all 144 MUL/MULHU and spills ~530 words to the stack
/// (measured: 1 870 instructions per 128-bit product instead of ~750).
#[inline(never)]
fn bmul32(x: u32, y: u32) -> u64 {
    let x0 = x & 0x1111_1111;
    let x1 = x & 0x2222_2222;
    let x2 = x & 0x4444_4444;
    let x3 = x & 0x8888_8888;
    let y0 = y & 0x1111_1111;
    let y1 = y & 0x2222_2222;
    let y2 = y & 0x4444_4444;
    let y3 = y & 0x8888_8888;
    #[inline(always)]
    fn m(a: u32, b: u32) -> u64 {
        (a as u64) * (b as u64)
    }
    let z0 = m(x0, y0) ^ m(x1, y3) ^ m(x2, y2) ^ m(x3, y1);
    let z1 = m(x0, y1) ^ m(x1, y0) ^ m(x2, y3) ^ m(x3, y2);
    let z2 = m(x0, y2) ^ m(x1, y1) ^ m(x2, y0) ^ m(x3, y3);
    let z3 = m(x0, y3) ^ m(x1, y2) ^ m(x2, y1) ^ m(x3, y0);
    (z0 & 0x1111_1111_1111_1111)
        | (z1 & 0x2222_2222_2222_2222)
        | (z2 & 0x4444_4444_4444_4444)
        | (z3 & 0x8888_8888_8888_8888)
}

/// 32x32 -> 64 carry-less product using 16 integer multiplications on operands
/// masked to every 4th bit. Each output position sums at most 8 one-bit
/// products, so the count fits in the 4-bit hole and never carries into the
/// next live position; its parity is the carry-less bit.
///
/// Inlined twin of [`bmul32`], for [`mul_ctkara_inl`].
#[inline(always)]
fn bmul32_inl(x: u32, y: u32) -> u64 {
    let x0 = x & 0x1111_1111;
    let x1 = x & 0x2222_2222;
    let x2 = x & 0x4444_4444;
    let x3 = x & 0x8888_8888;
    let y0 = y & 0x1111_1111;
    let y1 = y & 0x2222_2222;
    let y2 = y & 0x4444_4444;
    let y3 = y & 0x8888_8888;
    #[inline(always)]
    fn m(a: u32, b: u32) -> u64 {
        (a as u64) * (b as u64)
    }
    let z0 = m(x0, y0) ^ m(x1, y3) ^ m(x2, y2) ^ m(x3, y1);
    let z1 = m(x0, y1) ^ m(x1, y0) ^ m(x2, y3) ^ m(x3, y2);
    let z2 = m(x0, y2) ^ m(x1, y1) ^ m(x2, y0) ^ m(x3, y3);
    let z3 = m(x0, y3) ^ m(x1, y2) ^ m(x2, y1) ^ m(x3, y0);
    (z0 & 0x1111_1111_1111_1111)
        | (z1 & 0x2222_2222_2222_2222)
        | (z2 & 0x4444_4444_4444_4444)
        | (z3 & 0x8888_8888_8888_8888)
}

/// 64x64 -> 128 carry-less product, Karatsuba over 32-bit halves (3 bmul32).
#[inline(always)]
fn clmul64k(x0: u32, x1: u32, y0: u32, y1: u32) -> [u32; 4] {
    let lo = bmul32(x0, y0);
    let hi = bmul32(x1, y1);
    let mid = bmul32(x0 ^ x1, y0 ^ y1) ^ lo ^ hi;
    [
        lo as u32,
        ((lo >> 32) as u32) ^ (mid as u32),
        (hi as u32) ^ ((mid >> 32) as u32),
        (hi >> 32) as u32,
    ]
}

#[inline(never)]
pub fn mul_ctkara(a: u128, b: u128) -> u128 {
    let a = to_w(a);
    let b = to_w(b);
    let p0 = clmul64k(a[0], a[1], b[0], b[1]);
    let p2 = clmul64k(a[2], a[3], b[2], b[3]);
    let pm = clmul64k(a[0] ^ a[2], a[1] ^ a[3], b[0] ^ b[2], b[1] ^ b[3]);
    let p1 = [
        pm[0] ^ p0[0] ^ p2[0],
        pm[1] ^ p0[1] ^ p2[1],
        pm[2] ^ p0[2] ^ p2[2],
        pm[3] ^ p0[3] ^ p2[3],
    ];
    reduce8([
        p0[0],
        p0[1],
        p0[2] ^ p1[0],
        p0[3] ^ p1[1],
        p2[0] ^ p1[2],
        p2[1] ^ p1[3],
        p2[2],
        p2[3],
    ])
}

/// Same as [`clmul64k`] with the 32x32 products inlined.
#[inline(always)]
fn clmul64k_inl(x0: u32, x1: u32, y0: u32, y1: u32) -> [u32; 4] {
    let lo = bmul32_inl(x0, y0);
    let hi = bmul32_inl(x1, y1);
    let mid = bmul32_inl(x0 ^ x1, y0 ^ y1) ^ lo ^ hi;
    [
        lo as u32,
        ((lo >> 32) as u32) ^ (mid as u32),
        (hi as u32) ^ ((mid >> 32) as u32),
        (hi >> 32) as u32,
    ]
}

#[inline(never)]
pub fn mul_ctkara_inl(a: u128, b: u128) -> u128 {
    let a = to_w(a);
    let b = to_w(b);
    let p0 = clmul64k_inl(a[0], a[1], b[0], b[1]);
    let p2 = clmul64k_inl(a[2], a[3], b[2], b[3]);
    let pm = clmul64k_inl(a[0] ^ a[2], a[1] ^ a[3], b[0] ^ b[2], b[1] ^ b[3]);
    let p1 = [
        pm[0] ^ p0[0] ^ p2[0],
        pm[1] ^ p0[1] ^ p2[1],
        pm[2] ^ p0[2] ^ p2[2],
        pm[3] ^ p0[3] ^ p2[3],
    ];
    reduce8([
        p0[0],
        p0[1],
        p0[2] ^ p1[0],
        p0[3] ^ p1[1],
        p2[0] ^ p1[2],
        p2[1] ^ p1[3],
        p2[2],
        p2[3],
    ])
}

// ---------------------------------------------------------------------------
// Variant "ctkara64": 64x64 -> 128 carry-less products from the 64-bit
// integer multiply (MUL + MULHU on rv64) on operands masked to every 5th bit,
// one Karatsuba level (3 such products). Meant for 64-bit targets; on rv32 the
// u128 products become library calls.
// ---------------------------------------------------------------------------
const fn mask64(r: u32) -> u64 {
    let mut m = 0u64;
    let mut p = 0;
    while p < 64 {
        if p % 5 == r {
            m |= 1u64 << p;
        }
        p += 1;
    }
    m
}
const fn mask128(r: u32) -> u128 {
    let mut m = 0u128;
    let mut p = 0;
    while p < 128 {
        if p % 5 == r {
            m |= 1u128 << p;
        }
        p += 1;
    }
    m
}
const M64: [u64; 5] = [mask64(0), mask64(1), mask64(2), mask64(3), mask64(4)];
const M128: [u128; 5] = [mask128(0), mask128(1), mask128(2), mask128(3), mask128(4)];

/// 64x64 -> 128 carry-less product. Each masked operand has at most 13 bits,
/// so a single integer product sums at most 13 one-bit terms per position:
/// the count fits the 5-bit hole and never reaches the next live position.
/// The five products of one residue class are combined with XOR (not ADD),
/// which keeps every parity bit exact.
#[inline(always)]
fn bmul64(x: u64, y: u64) -> u128 {
    let xs = [x & M64[0], x & M64[1], x & M64[2], x & M64[3], x & M64[4]];
    let ys = [y & M64[0], y & M64[1], y & M64[2], y & M64[3], y & M64[4]];
    #[inline(always)]
    fn m(a: u64, b: u64) -> u128 {
        (a as u128) * (b as u128)
    }
    let mut z = 0u128;
    let mut r = 0;
    while r < 5 {
        let mut acc = 0u128;
        let mut i = 0;
        while i < 5 {
            let j = (r + 5 - i) % 5;
            acc ^= m(xs[i], ys[j]);
            i += 1;
        }
        z |= acc & M128[r];
        r += 1;
    }
    z
}

/// `ghash_reduce` of jetsam_ivc_core (64-bit words), copied.
#[inline(always)]
fn ghash_reduce64(r0: u64, r1: u64, r2: u64, r3: u64) -> u128 {
    let s1_lo = r2 << 1;
    let s1_hi = (r3 << 1) | (r2 >> 63);
    let s2_lo = r2 << 2;
    let s2_hi = (r3 << 2) | (r2 >> 62);
    let s7_lo = r2 << 7;
    let s7_hi = (r3 << 7) | (r2 >> 57);
    let t_lo = r2 ^ s1_lo ^ s2_lo ^ s7_lo;
    let t_hi = r3 ^ s1_hi ^ s2_hi ^ s7_hi;
    let ov = (r3 >> 63) ^ (r3 >> 62) ^ (r3 >> 57);
    let corr = ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
    ((r0 ^ t_lo ^ corr) as u128) | (((r1 ^ t_hi) as u128) << 64)
}

#[inline(never)]
pub fn mul_ctkara64(a: u128, b: u128) -> u128 {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);
    let lo = bmul64(a0, b0);
    let hi = bmul64(a1, b1);
    let mid = bmul64(a0 ^ a1, b0 ^ b1) ^ lo ^ hi;
    let r0 = lo as u64;
    let r1 = ((lo >> 64) as u64) ^ (mid as u64);
    let r2 = (hi as u64) ^ ((mid >> 64) as u64);
    let r3 = (hi >> 64) as u64;
    ghash_reduce64(r0, r1, r2, r3)
}

/// Squaring on 64-bit words: spread each byte with the same table.
#[inline(always)]
pub fn sq_spread64(a: u128) -> u128 {
    let mut r = [0u64; 4];
    for w in 0..2 {
        let x = (a >> (64 * w)) as u64;
        let mut lo = 0u64;
        let mut hi = 0u64;
        for k in 0..4 {
            lo |= (SPREAD8[((x >> (8 * k)) & 0xff) as usize] as u64) << (16 * k);
            hi |= (SPREAD8[((x >> (32 + 8 * k)) & 0xff) as usize] as u64) << (16 * k);
        }
        r[2 * w] = lo;
        r[2 * w + 1] = hi;
    }
    ghash_reduce64(r[0], r[1], r[2], r[3])
}

// ---------------------------------------------------------------------------
// Variant markers.
// ---------------------------------------------------------------------------
pub struct Repo;
impl Gf for Repo {
    const NAME: &'static str = "repo-soft (verbatim u128 fallback)";
    #[inline(always)]
    fn mul(a: u128, b: u128) -> u128 {
        repo::clmul_block128(a, b)
    }
    #[inline(always)]
    fn sq(a: u128) -> u128 {
        repo::square_flat_u128(a)
    }
}

pub struct Naive;
impl Gf for Naive {
    const NAME: &'static str = "naive shift-xor";
    #[inline(always)]
    fn mul(a: u128, b: u128) -> u128 {
        mul_naive(a, b)
    }
    #[inline(always)]
    fn sq(a: u128) -> u128 {
        sq_spread(a)
    }
}

pub struct Comb4;
impl Gf for Comb4 {
    const NAME: &'static str = "comb 4-bit table";
    #[inline(always)]
    fn mul(a: u128, b: u128) -> u128 {
        mul_comb4(a, b)
    }
    #[inline(always)]
    fn sq(a: u128) -> u128 {
        sq_spread(a)
    }
}

pub struct CtKara64;
impl Gf for CtKara64 {
    const NAME: &'static str = "ctmul64-karatsuba (64x64 MUL/MULHU, holes every 5 bits)";
    #[inline(always)]
    fn mul(a: u128, b: u128) -> u128 {
        mul_ctkara64(a, b)
    }
    #[inline(always)]
    fn sq(a: u128) -> u128 {
        sq_spread64(a)
    }
}

pub struct CtKaraInl;
impl Gf for CtKaraInl {
    const NAME: &'static str = "ctmul-karatsuba, 32x32 products inlined";
    #[inline(always)]
    fn mul(a: u128, b: u128) -> u128 {
        mul_ctkara_inl(a, b)
    }
    #[inline(always)]
    fn sq(a: u128) -> u128 {
        sq_spread(a)
    }
}

pub struct CtKara;
impl Gf for CtKara {
    const NAME: &'static str = "ctmul-karatsuba (integer MUL with holes)";
    #[inline(always)]
    fn mul(a: u128, b: u128) -> u128 {
        mul_ctkara(a, b)
    }
    #[inline(always)]
    fn sq(a: u128) -> u128 {
        sq_spread(a)
    }
}

/// Run `n` dependent multiplications: x <- x * y ^ K1, y <- y ^ x (both
/// operands change every step, nothing can be hoisted or skipped).
#[inline(never)]
pub fn mul_chain<G: Gf>(mut x: u128, mut y: u128, n: u32) -> u128 {
    const K1: u128 = 0x9E37_79B9_7F4A_7C15_F39C_C060_5CED_C834;
    let mut i = 0;
    while i < n {
        x = G::mul(x, y) ^ K1;
        y ^= x;
        i += 1;
    }
    x ^ y
}

/// Same loop with the multiplication replaced by XOR: measures the chaining
/// glue so it can be subtracted.
pub struct XorOnly;
impl Gf for XorOnly {
    const NAME: &'static str = "baseline (xor instead of mul)";
    #[inline(never)]
    fn mul(a: u128, b: u128) -> u128 {
        a ^ b
    }
    #[inline(always)]
    fn sq(a: u128) -> u128 {
        a
    }
}
