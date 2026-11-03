// SPDX-License-Identifier: Apache-2.0
//! keccak256 (original Keccak padding 0x01, as the EVM KECCAK256 opcode) over a pluggable keccak-f[1600]:
//! the SP1 guest passes its standard KECCAK_PERMUTE precompile, the native
//! gate the software permutation below.
pub type PermuteFn = fn(&mut [u64; 25]);

const RC: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
];
const RHO: [u32; 24] = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44];
const PI: [usize; 24] = [10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1];

pub fn keccak_f_soft(a: &mut [u64; 25]) {
    for rc in RC.iter() {
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        let mut last = a[1];
        for i in 0..24 {
            let j = PI[i];
            let t = a[j];
            a[j] = last.rotate_left(RHO[i]);
            last = t;
        }
        for y in 0..5 {
            let mut row = [0u64; 5];
            row.copy_from_slice(&a[5 * y..5 * y + 5]);
            for x in 0..5 {
                a[5 * y + x] = row[x] ^ ((!row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }
        a[0] ^= rc;
    }
}

/// Streaming keccak256 (rate 136).
pub struct Keccak {
    st: [u64; 25],
    buf: [u8; 136],
    n: usize,
    f: PermuteFn,
}
impl Keccak {
    pub fn new(f: PermuteFn) -> Self {
        Self { st: [0; 25], buf: [0; 136], n: 0, f }
    }
    fn absorb_block(&mut self) {
        for i in 0..17 {
            let mut w = [0u8; 8];
            w.copy_from_slice(&self.buf[i * 8..i * 8 + 8]);
            self.st[i] ^= u64::from_le_bytes(w);
        }
        (self.f)(&mut self.st);
        self.n = 0;
    }
    pub fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            let k = core::cmp::min(136 - self.n, data.len());
            self.buf[self.n..self.n + k].copy_from_slice(&data[..k]);
            self.n += k;
            data = &data[k..];
            if self.n == 136 {
                self.absorb_block();
            }
        }
    }
    pub fn finalize(mut self) -> [u8; 32] {
        for b in self.buf[self.n..].iter_mut() {
            *b = 0;
        }
        self.buf[self.n] ^= 0x01;
        self.buf[135] ^= 0x80;
        self.absorb_block();
        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..i * 8 + 8].copy_from_slice(&self.st[i].to_le_bytes());
        }
        out
    }
}
pub fn keccak256(f: PermuteFn, data: &[u8]) -> [u8; 32] {
    let mut k = Keccak::new(f);
    k.update(data);
    k.finalize()
}
