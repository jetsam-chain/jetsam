// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! SHA3-256 (FIPS 202), only for the two-byte checksum of v3 onion addresses
//! (rend-spec-v3 section 6). Not used for anything security-bearing: a wrong
//! checksum only means an address is refused before Tor would refuse it.

const RATE: usize = 136;

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

const ROTATIONS: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

const PI_LANES: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

fn keccak_f1600(state: &mut [u64; 25]) {
    for round_constant in ROUND_CONSTANTS {
        // theta
        let mut columns = [0u64; 5];
        for (x, column) in columns.iter_mut().enumerate() {
            *column = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for x in 0..5 {
            let mix = columns[(x + 4) % 5] ^ columns[(x + 1) % 5].rotate_left(1);
            for y in (0..25).step_by(5) {
                state[y + x] ^= mix;
            }
        }
        // rho and pi
        let mut carried = state[1];
        for (lane, rotation) in PI_LANES.iter().zip(ROTATIONS) {
            let next = state[*lane];
            state[*lane] = carried.rotate_left(rotation);
            carried = next;
        }
        // chi
        for y in (0..25).step_by(5) {
            let row = [
                state[y],
                state[y + 1],
                state[y + 2],
                state[y + 3],
                state[y + 4],
            ];
            for x in 0..5 {
                state[y + x] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
            }
        }
        // iota
        state[0] ^= round_constant;
    }
}

fn absorb_block(state: &mut [u64; 25], block: &[u8; RATE]) {
    for (lane, bytes) in state.iter_mut().zip(block.chunks_exact(8)) {
        let mut word = [0u8; 8];
        word.copy_from_slice(bytes);
        *lane ^= u64::from_le_bytes(word);
    }
    keccak_f1600(state);
}

pub(crate) fn sha3_256(data: &[u8]) -> [u8; 32] {
    let mut state = [0u64; 25];
    let mut blocks = data.chunks_exact(RATE);
    for block in &mut blocks {
        let mut full = [0u8; RATE];
        full.copy_from_slice(block);
        absorb_block(&mut state, &full);
    }
    let tail = blocks.remainder();
    let mut last = [0u8; RATE];
    last[..tail.len()].copy_from_slice(tail);
    // SHA-3 domain separation (01) and pad10*1.
    last[tail.len()] ^= 0x06;
    last[RATE - 1] ^= 0x80;
    absorb_block(&mut state, &last);

    let mut digest = [0u8; 32];
    for (out, lane) in digest.chunks_exact_mut(8).zip(state.iter()) {
        out.copy_from_slice(&lane.to_le_bytes());
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    // Vectors computed with Python's hashlib.sha3_256.
    #[test]
    fn sha3_256_matches_the_fips_202_vectors() {
        assert_eq!(
            hex(&sha3_256(b"")),
            "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"
        );
        assert_eq!(
            hex(&sha3_256(b"abc")),
            "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"
        );
        let counting: Vec<u8> = (0u8..200).collect();
        assert_eq!(
            hex(&sha3_256(&counting)),
            "5f728f63bf5ee48c77f453c0490398fa645b8d4c4e56be9a41cfec344d6ca899"
        );
    }

    #[test]
    fn sha3_256_pads_correctly_around_the_rate_boundary() {
        assert_eq!(
            hex(&sha3_256(&[b'a'; 135])),
            "8094bb53c44cfb1e67b7c30447f9a1c33696d2463ecc1d9c92538913392843c9"
        );
        assert_eq!(
            hex(&sha3_256(&[b'a'; 136])),
            "3fc5559f14db8e453a0a3091edbd2bc25e11528d81c66fa570a4efdcc2695ee1"
        );
        assert_eq!(
            hex(&sha3_256(&[b'a'; 137])),
            "f8d6846cedd2ccfadf15c5879ef95af724d799eed7391fb1c91f95344e738614"
        );
    }
}
