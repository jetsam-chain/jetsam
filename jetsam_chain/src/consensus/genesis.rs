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
#[cfg(not(feature = "testnet"))]
pub const GENESIS_TIMESTAMP: u64 = 1_787_328_000;

/// Test-chain genesis timestamp (2026-09-26 00:00:00 UTC).
///
/// A different genesis is what actually keeps the two networks apart: it
/// changes the genesis hash, therefore the network `profile_id`, and two nodes
/// with different profiles refuse each other **before** any block is offered.
/// The ticker and the address HRP protect humans; this protects the protocol.
///
/// # Why this value moved on 2026-09-26 (it was `1_788_912_000`, 2026-09-09)
///
/// The v1.4 proof-of-work fork is rehearsed on this chain from a clean start, and
/// a clean start is not a directory somebody emptied. Keeping the old genesis
/// would leave every node that still holds the 7477-block history able to offer
/// it, and branch choice on cumulative work would bury the new chain under it.
/// With a new genesis, `refuse_foreign_chain_data` (`jetsam_node/src/main.rs`)
/// reads the genesis header out of the old `mdbx.dat` and refuses to run against
/// it at all — the old chain becomes unreachable rather than merely unwanted, and
/// nothing has to be deleted to make that true.
///
/// Moving the timestamp is the smallest edit that moves the genesis id: every
/// other field of `genesis_header()` is either structurally fixed (a zero
/// previous hash, the empty-state root, height 0) or shared with the public
/// network, which must not move.
#[cfg(feature = "testnet")]
pub const GENESIS_TIMESTAMP: u64 = 1_791_385_200;

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
#[cfg(not(feature = "testnet"))]
const GENESIS_NONCE: u128 = 131_160;

/// Test-chain genesis nonce, re-mined for the test-chain timestamp.
/// `genesis_nonce_satisfies_pow` below proves it satisfies the target, whichever
/// profile is compiled.
///
/// Re-mined on 2026-09-26 for `GENESIS_TIMESTAMP = 1_790_380_800` (it was
/// `300_173`, for the 2026-09-09 timestamp). The nonce is a function of the
/// header, so moving the timestamp invalidates it: `genesis_nonce_satisfies_pow`
/// is what refuses a build whose genesis cannot be proved.
#[cfg(feature = "testnet")]
const GENESIS_NONCE: u128 = 91_696;

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

    /// The mainnet genesis id, anchored. This is the chain that exists; the
    /// value must never move again.
    #[test]
    #[cfg(not(feature = "testnet"))]
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

    /// The test chain's genesis id, anchored the same way — and required to
    /// differ from the mainnet one. That difference is the whole separation:
    /// it changes the network `profile_id`, so a testnet node and a mainnet
    /// node refuse each other at the handshake instead of exchanging blocks.
    ///
    /// It also differs from the id the **previous** test chain carried
    /// (`b3efb3c1…996d`, genesis timestamp 2026-09-09), and that is asserted here
    /// too: the whole point of the 2026-09-26 reset is that a node built from this
    /// source cannot be talked into the old history, and a silent revert of the
    /// timestamp would undo it without any other test noticing.
    #[test]
    #[cfg(feature = "testnet")]
    fn testnet_genesis_block_id_is_canonical_and_differs_from_mainnet() {
        const MAINNET_GENESIS_ID: [u8; 32] = [
            0x6e, 0x59, 0x2c, 0x07, 0xbe, 0x6f, 0xd1, 0xb4, 0x25, 0x9e, 0xea, 0xcb, 0xf4, 0xeb,
            0x7e, 0xb2, 0x94, 0x8a, 0x77, 0xf1, 0xd0, 0x26, 0x26, 0xa1, 0x2f, 0xda, 0xb4, 0x2c,
            0x44, 0x8c, 0x5f, 0x44,
        ];
        /// The chain that was reset on 2026-09-26 at height 7477. Its data
        /// directories still exist, moved aside; this build must never agree with
        /// them.
        const RETIRED_TESTNET_GENESIS_ID: [u8; 32] = [
            0xb3, 0xef, 0xb3, 0xc1, 0xd3, 0x1f, 0xee, 0x8b, 0x9a, 0xee, 0x7b, 0x06, 0xcb, 0x11,
            0x2f, 0xae, 0xa5, 0xab, 0xc1, 0xce, 0xb7, 0x35, 0xd2, 0x1c, 0xa5, 0xa3, 0x90, 0x1b,
            0x11, 0x0f, 0x99, 0x6d,
        ];
        let id = crate::block_header::block_id(&genesis_header());
        assert_ne!(id, MAINNET_GENESIS_ID, "the two chains must not share a genesis");
        assert_ne!(
            id, RETIRED_TESTNET_GENESIS_ID,
            "the reset test chain must not share a genesis with the one it replaced"
        );
        /// Retired: 2026-10-07 J=1920 rehearsal born in v1.5, stopped at height 77.
        const RETIRED_BC341146: [u8; 32] = [
            0xbc, 0x34, 0x11, 0x46, 0xcb, 0x97, 0x4c, 0xad, 0x51, 0xd5, 0x39, 0x8f, 0x00, 0x97,
            0xc9, 0x40, 0x07, 0xbf, 0xf7, 0x3d, 0xae, 0x59, 0x2c, 0xb8, 0x0e, 0xfd, 0x4f, 0x01,
            0xcf, 0x11, 0xcf, 0x87,
        ];
        assert_ne!(crate::block_header::block_id(&genesis_header()), RETIRED_BC341146);
        /// Retired: 2026-10-04 J=5980 rehearsal, stopped at height 7664.
        const RETIRED_B3D4220C: [u8; 32] = [
            0xb3, 0xd4, 0x22, 0x0c, 0xe6, 0xdb, 0xb2, 0xa8, 0xda, 0x03, 0xd7, 0xcc, 0xc9, 0x69,
            0x5b, 0x24, 0xb1, 0xc8, 0xe6, 0xc7, 0xd8, 0x49, 0x30, 0x88, 0x36, 0x93, 0xce, 0x63,
            0xcc, 0xb4, 0x55, 0xf8,
        ];
        assert_ne!(crate::block_header::block_id(&genesis_header()), RETIRED_B3D4220C);
        const TESTNET_GENESIS_ID: [u8; 32] = [
            0xd9, 0xd1, 0x56, 0xee, 0x35, 0xe7, 0x25, 0xcf, 0x6b, 0x13, 0x32, 0x16, 0x80, 0x9c,
            0x1b, 0x0f, 0x16, 0x5d, 0x11, 0x20, 0xc9, 0xcf, 0x42, 0xcc, 0x2b, 0xfe, 0xb0, 0xef,
            0x90, 0x31, 0xee, 0x85,
        ];
        assert_eq!(id, TESTNET_GENESIS_ID);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn genesis_timestamp_is_reasonable() {
        #[cfg(not(feature = "testnet"))]
        assert_eq!(GENESIS_TIMESTAMP, 1_787_328_000);
        // 2026-10-07 15:00:00 UTC — the reset test chain (proof of work walked
        // from block 6). Retired values: 1_788_912_000, 1_790_380_800.
        #[cfg(feature = "testnet")]
        assert_eq!(GENESIS_TIMESTAMP, 1_791_385_200);
    }
}
