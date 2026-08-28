// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Genesis block construction.
//!
//! The genesis block is hardcoded: same bytes on every node.
//! It has:
//! - Zero state (all slots empty)
//! - Fixed Poseidon2b PoW target
//! - Coinbase to burn address (initial coins bootstrapping)
//! - no transported HistoryStep terminal; genesis is the built-in recursion boundary
//!
//! The genesis state_root is the direct exact sparse-Merkle root of an empty UTXO tree.

use crate::block_header::BlockHeader;
use crate::consensus::{
    params::{GENESIS_TARGET, LOG_SLOTS_GENESIS},
    pow::search_pow,
};
use jetsam_poseidon2b::primitives::Address;

/// Fixed mainnet genesis timestamp (2026-08-21 16:00:00 UTC).
pub const GENESIS_TIMESTAMP: u64 = 1_787_328_000;

/// The genesis burn address — coinbase recipient at height 0.
/// Uses a zero address; no private key is known.
pub const GENESIS_BURN_ADDRESS: Address = Address([0u8; 32]);

/// Build the canonical genesis block header.
///
/// The header's PoW is pre-computed and hardcoded. The `state_root` is the
/// canonical empty-state root and `tx_root` is all-zeros (coinbase-only,
/// computed by the full node layer).
///
/// Every node must produce byte-identical output from this function.
pub fn genesis_header() -> BlockHeader {
    BlockHeader {
        prev_block_hash: [0u8; 32],
        state_root: genesis_state_root(),
        tx_root: [0u8; 32],
        timestamp: GENESIS_TIMESTAMP,
        height: 0,
        miner_address: GENESIS_BURN_ADDRESS,
        nonce: GENESIS_NONCE,
        difficulty_target: GENESIS_TARGET,
        // Genesis is built in and has no attached HistoryStep terminal.
        log_slots: LOG_SLOTS_GENESIS,
        active_slot_count: 0,
        alloc_counter: 0,
    }
}

/// The canonical genesis state root of the all-zero exact UTXO tree at
/// `LOG_SLOTS_GENESIS`.
///
/// Computed from `zero_slot_roots(LOG_SLOTS_GENESIS)` and hardcoded.
/// Verified by the test `genesis_state_root_matches_computed` below.
pub fn genesis_state_root() -> [u8; 32] {
    GENESIS_STATE_ROOT
}

/// Pre-computed genesis state root. All 2^24 slots are zero.
// JETSAM CHANGE: recomputed. The TowerHash round constants and domain tags
// changed, so every hash in the chain changed with them.
const GENESIS_STATE_ROOT: [u8; 32] = [
    0x02, 0x19, 0xf9, 0xd6, 0x5c, 0x10, 0x64, 0xbf, 0xa4, 0x78, 0x36, 0x2e, 0x7b, 0x6f, 0xcd, 0xce,
    0xf5, 0xa2, 0x99, 0x3f, 0x5d, 0x44, 0x27, 0x7a, 0x74, 0x30, 0x95, 0xc4, 0xfc, 0xe7, 0xe6, 0xdb,
];

/// Pre-mined genesis nonce.
/// Satisfies: `H_POSEIDON_POW(genesis_header()) < GENESIS_TARGET`.
/// Mined for the canonical 16-field PoW schedule.
// JETSAM CHANGE: re-mined against the new TowerHash schedule.
const GENESIS_NONCE: u128 = 131_160;

/// Find and return a valid genesis nonce at runtime.
/// Used for verification only — not for production (nonce is hardcoded as `GENESIS_NONCE`).
pub fn find_genesis_nonce() -> u128 {
    let mut h = genesis_header();
    h.nonce = 0;
    search_pow(&h, 0, 100_000_000).expect("genesis target is trivially satisfiable")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genesis_header_is_deterministic() {
        let a = genesis_header();
        let b = genesis_header();
        assert_eq!(a.height, b.height);
        assert_eq!(a.timestamp, b.timestamp);
        assert_eq!(a.difficulty_target, b.difficulty_target);
        assert_eq!(a.prev_block_hash, b.prev_block_hash);
    }

    #[test]
    fn genesis_header_fields() {
        let h = genesis_header();
        assert_eq!(h.height, 0);
        assert_eq!(h.prev_block_hash, [0u8; 32]);
        assert_eq!(h.difficulty_target, GENESIS_TARGET);
        assert_eq!(h.log_slots, LOG_SLOTS_GENESIS);
        assert_eq!(h.active_slot_count, 0);
        assert_eq!(h.alloc_counter, 0);
    }

    #[test]
    fn genesis_state_root_matches_computed() {
        let mut state = crate::state::ChainState::with_log_slots(24);
        assert_eq!(state.state_root(), genesis_state_root());
    }

    /// Print the new genesis state root and a valid nonce for it.
    /// Run with: cargo test -p jetsam_chain --lib -- consensus::genesis::tests::print_new_genesis --nocapture
    #[test]
    #[ignore]
    fn print_new_genesis() {
        let mut state = crate::state::ChainState::with_log_slots(24);
        let new_root = state.state_root();
        println!("\nNew GENESIS_STATE_ROOT:");
        print!("const GENESIS_STATE_ROOT: [u8; 32] = [");
        for (i, b) in new_root.iter().enumerate() {
            if i % 16 == 0 {
                print!("\n    ");
            }
            print!("0x{:02x}, ", b);
        }
        println!("\n];");
        let new_nonce = find_genesis_nonce_for(&new_root);
        println!("New GENESIS_NONCE: {}", new_nonce);
        print!("New GENESIS_BLOCK_ID: ");
        for byte in crate::block_header::block_id(&genesis_header()) {
            print!("{byte:02x}");
        }
        println!();
    }

    fn find_genesis_nonce_for(state_root: &[u8; 32]) -> u128 {
        use crate::block_header::BlockHeader;
        use crate::consensus::params::{GENESIS_TARGET, LOG_SLOTS_GENESIS};
        use crate::consensus::pow::search_pow;
        let h = BlockHeader {
            prev_block_hash: [0u8; 32],
            state_root: *state_root,
            tx_root: [0u8; 32],
            timestamp: GENESIS_TIMESTAMP,
            height: 0,
            miner_address: GENESIS_BURN_ADDRESS,
            nonce: 0,
            difficulty_target: GENESIS_TARGET,
            log_slots: LOG_SLOTS_GENESIS,
            active_slot_count: 0,
            alloc_counter: 0,
        };
        search_pow(&h, 0, 2_000_000_000).expect("genesis target is trivially satisfiable")
    }

    #[test]
    fn genesis_nonce_satisfies_pow() {
        let h = genesis_header();
        use crate::consensus::pow::validate_pow;
        assert!(
            validate_pow(&h).is_ok(),
            "GENESIS_NONCE={} must satisfy PoW",
            GENESIS_NONCE
        );
    }

    #[test]
    fn genesis_block_id_is_canonical() {
        assert_eq!(
            crate::block_header::block_id(&genesis_header()),
[
                0x6e, 0x59, 0x2c, 0x07, 0xbe, 0x6f, 0xd1, 0xb4, 0x25, 0x9e, 0xea, 0xcb, 0xf4, 0xeb,
                0x7e, 0xb2, 0x94, 0x8a, 0x77, 0xf1, 0xd0, 0x26, 0x26, 0xa1, 0x2f, 0xda, 0xb4, 0x2c,
                0x44, 0x8c, 0x5f, 0x44,
            ]
        );
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn genesis_timestamp_is_reasonable() {
        assert_eq!(GENESIS_TIMESTAMP, 1_787_328_000);
    }
}
