// SPDX-License-Identifier: Apache-2.0
//! Poseidon2b sponge (t=4, rate=2) and 2-to-1 compression, tower basis,
//! as `jetsam_poseidon2b::native::compression` (the bridge proof program's port, moved here with the
//! logical txid it hashes, A5-12: [`crate::txid`]).
use tw_core::gf::CtKara64;
use tw_core::header::permute_tower;

pub const fn tag(label: &[u8; 8]) -> u64 {
    u64::from_be_bytes(*label)
}
pub const TAG_BLOCKHDR: u64 = tag(b"E_BLKHDR");
pub const TAG_COMPRESS: u64 = tag(b"E_COMPRS");
pub const TAG_TXROOT: u64 = tag(b"E_TXROOT");
pub const TAG_TX8X2: u64 = tag(b"E_TX8X2_");
pub const TAG_PAGED_SPEND: u64 = tag(b"PAGEDTX_");

#[inline(always)]
pub fn capacity_iv(t: u64) -> [u128; 2] {
    [(t as u128) << 64, t as u128]
}

/// `Poseidon2bPermutation::permute_mut` (tower -> flat -> permute -> tower).
#[inline(always)]
pub fn perm(st: &mut [u128; 4]) {
    permute_tower::<CtKara64>(st)
}

#[inline(always)]
fn u128_at(b: &[u8], o: usize) -> u128 {
    let mut x = [0u8; 16];
    x.copy_from_slice(&b[o..o + 16]);
    u128::from_le_bytes(x)
}

#[inline(always)]
fn out32(st: &[u128; 4]) -> [u8; 32] {
    let mut o = [0u8; 32];
    o[..16].copy_from_slice(&st[0].to_le_bytes());
    o[16..].copy_from_slice(&st[1].to_le_bytes());
    o
}

pub struct Sponge {
    st: [u128; 4],
    buf: [u8; 32],
    filled: usize,
}

impl Sponge {
    pub fn with_iv(t: u64) -> Self {
        let iv = capacity_iv(t);
        Self { st: [0, 0, iv[0], iv[1]], buf: [0u8; 32], filled: 0 }
    }
    fn permute_buffer(&mut self) {
        self.st[0] ^= u128_at(&self.buf, 0);
        self.st[1] ^= u128_at(&self.buf, 16);
        perm(&mut self.st);
    }
    pub fn update(&mut self, mut data: &[u8]) {
        if self.filled != 0 {
            let n = core::cmp::min(data.len(), 32 - self.filled);
            self.buf[self.filled..self.filled + n].copy_from_slice(&data[..n]);
            data = &data[n..];
            self.filled += n;
            if self.filled == 32 {
                self.permute_buffer();
                self.filled = 0;
            }
        }
        let mut chunks = data.chunks_exact(32);
        for c in &mut chunks {
            self.buf.copy_from_slice(c);
            self.permute_buffer();
        }
        let rem = chunks.remainder();
        if !rem.is_empty() {
            self.buf[..rem.len()].copy_from_slice(rem);
            self.filled = rem.len();
        }
    }
    /// `absorb(Block128::from(x))`: 16 little-endian bytes.
    pub fn absorb_u128(&mut self, x: u128) {
        self.update(&x.to_le_bytes());
    }
    /// `absorb_pair(lo16, hi16)` of a 32-byte digest = the digest bytes.
    pub fn absorb_digest(&mut self, d: &[u8]) {
        self.update(&d[..32]);
    }
    /// `finalize`: `0x80..01` padding over the rest of the block, one permutation.
    pub fn finalize(mut self) -> [u8; 32] {
        for b in self.buf[self.filled..].iter_mut() {
            *b = 0;
        }
        self.buf[self.filled] |= 0x80;
        self.buf[31] |= 0x01;
        self.permute_buffer();
        out32(&self.st)
    }
}

/// `compress_with_tag(tag, a, b)`: two permutations.
pub fn compress_with_tag(t: u64, a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let iv = capacity_iv(t);
    let mut st = [u128_at(a, 0), u128_at(a, 16), iv[0], iv[1]];
    perm(&mut st);
    st[0] ^= u128_at(b, 0);
    st[1] ^= u128_at(b, 16);
    perm(&mut st);
    out32(&st)
}

/// `compress` (`TAG_COMPRESS`).
#[inline]
pub fn compress(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    compress_with_tag(TAG_COMPRESS, a, b)
}

/// One-permutation wrap under a tag (`tx8x2_wrap`).
pub fn wrap1(t: u64, r: &[u8; 32]) -> [u8; 32] {
    let iv = capacity_iv(t);
    let mut st = [u128_at(r, 0), u128_at(r, 16), iv[0], iv[1]];
    perm(&mut st);
    out32(&st)
}
