// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! TEST material only: synthetic headers and chains, the T0.5 TEST dealing.
//! Never a real key, never a real chain.

use jetsam_chain::block_header::BlockHeader;
use jetsam_poseidon2b::primitives::Address;

use crate::chain::{Chain, HeaderInfo};

/// An easy target whose work is `2^256 / 2^k`-ish: target = 2^(256−k) − 1.
pub fn target_bits(k: u32) -> [u8; 32] {
    // little-endian: the integer 2^(256-k) - 1
    let mut out = [0u8; 32];
    let bits = 256 - k;
    for b in 0..bits {
        out[(b / 8) as usize] |= 1 << (b % 8);
    }
    out
}

/// A synthetic header on top of `prev` (id), at `height`, time `ts`.
pub fn header(prev: [u8; 32], height: u64, ts: u64, target: [u8; 32], salt: u8) -> HeaderInfo {
    let mut tx_root = [0u8; 32];
    tx_root[0] = salt;
    tx_root[1..9].copy_from_slice(&height.to_le_bytes());
    let h = BlockHeader {
        prev_block_hash: prev,
        state_root: [salt; 32],
        tx_root,
        timestamp: ts,
        height,
        miner_address: Address([7u8; 32]),
        nonce: u128::from(salt),
        difficulty_target: target,
        log_slots: 16,
        active_slot_count: height,
        alloc_counter: height,
    };
    HeaderInfo::from_header(&h)
}

/// A genesis header.
pub fn genesis() -> HeaderInfo {
    header([0u8; 32], 0, 1_700_000_000, target_bits(20), 0)
}

/// A chain from `genesis()` with `n` blocks of target `target_bits(20)`,
/// 180 s apart, branch `salt`.
pub fn chain(n: u64, salt: u8) -> Chain {
    let mut c = Chain::from_genesis(genesis());
    extend(&mut c, n, salt, 20);
    c
}

/// Extend `c` by `n` blocks of target `target_bits(bits)`, 180 s apart.
pub fn extend(c: &mut Chain, n: u64, salt: u8, bits: u32) {
    for _ in 0..n {
        let h = c.tip_height() + 1;
        let ts = 1_700_000_000 + 180 * h;
        c.extend(header(c.tip_id(), h, ts, target_bits(bits), salt))
            .expect("a synthetic header extends");
    }
}

/// The headers of `c` from `from` to `to`, hex.
pub fn raw_range(c: &Chain, from: u64, to: u64) -> Vec<String> {
    (from..=to)
        .map(|h| c.header_at(h).expect("header").raw_hex())
        .collect()
}
