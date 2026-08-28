// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Domain-separated exact-state commitment hashes.
//!
//! These helpers are intentionally independent from the raw segment storage
//! commitment path. They are pure building blocks for the exact Merkle state
//! transition proof.

#[cfg(test)]
use jetsam_core::Block128;
use jetsam_poseidon2b::native::compression::{compress_with_tag, Poseidon2bSponge};
use jetsam_poseidon2b::native::domain::{capacity_iv, TAG_EXSTNOD, TAG_EXSTSLT};

use crate::fri_state::SlotValue;

/// 32-byte exact-state hash.
pub type StateHash = [u8; 32];

/// Hash one exact-state slot leaf.
///
/// The full packed `(amount, creation_id)` lane is absorbed. Every 128-bit
/// lane is a canonical pair of `u64`s, so hashing is infallible.
pub fn slot_leaf_hash(slot: SlotValue) -> StateHash {
    let mut s = Poseidon2bSponge::with_iv(capacity_iv(TAG_EXSTSLT));
    s.absorb(slot.value);
    s.absorb_pair(slot.owner_hi, slot.owner_lo);
    s.finalize()
}

/// Hash one exact-state binary Merkle node.
pub fn state_node_hash(left: StateHash, right: StateHash) -> StateHash {
    compress_with_tag(TAG_EXSTNOD, &left, &right)
}

/// Precompute `Z_0..=Z_max_depth` for the exact sparse UTXO tree.
pub fn zero_slot_roots(max_depth: usize) -> Vec<StateHash> {
    let mut roots = Vec::with_capacity(max_depth.saturating_add(1));
    roots.push(slot_leaf_hash(SlotValue::EMPTY));
    for depth in 0..max_depth {
        let z = roots[depth];
        roots.push(state_node_hash(z, z));
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;
    use jetsam_core::TowerField;

    fn slot(value: u128, hi: u128, lo: u128) -> SlotValue {
        SlotValue {
            value: Block128::from(value),
            owner_hi: Block128::from(hi),
            owner_lo: Block128::from(lo),
        }
    }

    #[test]
    fn slot_hash_binds_full_packed_value_and_zero_id_uses_low_lane() {
        let s = slot(42, 7, 9);
        assert_eq!(slot_leaf_hash(s), slot_leaf_hash(s));
        let incarnation = slot((1u128 << 64) | 42, 7, 9);
        assert_ne!(slot_leaf_hash(s), slot_leaf_hash(incarnation));

        let mut zero_id = Poseidon2bSponge::with_iv(capacity_iv(TAG_EXSTSLT));
        zero_id.absorb(Block128::from(42u64));
        zero_id.absorb_pair(Block128::from(7u64), Block128::from(9u64));
        assert_eq!(slot_leaf_hash(s), zero_id.finalize());
    }

    #[test]
    fn node_order_and_domains_matter() {
        let a = slot_leaf_hash(slot(1, 2, 3));
        let b = slot_leaf_hash(slot(4, 5, 6));
        assert_ne!(state_node_hash(a, b), state_node_hash(b, a));
        assert_ne!(slot_leaf_hash(SlotValue::EMPTY), state_node_hash(a, b));
    }

    #[test]
    fn zero_roots_match_explicit_empty_trees() {
        let roots = zero_slot_roots(5);
        assert_eq!(roots.len(), 6);
        let mut root = slot_leaf_hash(SlotValue::EMPTY);
        assert_eq!(root, roots[0]);
        for &expected in roots.iter().take(6).skip(1) {
            root = state_node_hash(root, root);
            assert_eq!(root, expected);
        }
    }

    #[test]
    /// JETSAM helper: reprint the golden vectors after any change to the hash
    /// (round constants, domain tags, permutation). Run with
    /// `cargo test --release -p jetsam_chain --lib print_zero_root_golden -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn print_zero_root_golden() {
        let roots = zero_slot_roots(32);
        for depth in [0usize, 1, 4, 16, 24, 32] {
            println!("            (");
            println!("                {depth}usize,");
            println!("                [");
            let line: Vec<String> = roots[depth].iter().map(|b| b.to_string()).collect();
            for chunk in line.chunks(14) {
                println!("                    {},", chunk.join(", "));
            }
            println!("                ],");
            println!("            ),");
        }
    }

    fn zero_root_golden_vectors_are_stable() {
        let roots = zero_slot_roots(32);
        // JETSAM CHANGE: recomputed. New round constants and domain tags
        // change every digest the chain produces. Regenerate with
        // print_zero_root_golden above.
        let vectors = [
            (
                0usize,
                [
                    169, 62, 71, 162, 109, 95, 87, 15, 82, 47, 191, 147, 103, 5,
                    135, 123, 202, 57, 21, 37, 214, 223, 92, 14, 95, 139, 106, 223,
                    31, 190, 31, 147,
                ],
            ),
            (
                1usize,
                [
                    16, 93, 153, 119, 195, 207, 76, 201, 81, 214, 64, 20, 243, 115,
                    127, 49, 93, 97, 42, 44, 137, 193, 215, 23, 74, 250, 96, 218,
                    95, 163, 209, 215,
                ],
            ),
            (
                4usize,
                [
                    42, 32, 73, 21, 194, 100, 136, 30, 76, 68, 119, 187, 67, 210,
                    123, 96, 62, 189, 135, 158, 31, 98, 178, 77, 12, 252, 43, 74,
                    159, 209, 1, 50,
                ],
            ),
            (
                16usize,
                [
                    59, 187, 62, 33, 56, 73, 182, 33, 254, 181, 162, 187, 4, 61,
                    226, 23, 72, 200, 9, 76, 243, 23, 7, 225, 123, 185, 47, 248,
                    34, 179, 4, 182,
                ],
            ),
            (
                24usize,
                [
                    2, 25, 249, 214, 92, 16, 100, 191, 164, 120, 54, 46, 123, 111,
                    205, 206, 245, 162, 153, 63, 93, 68, 39, 122, 116, 48, 149, 196,
                    252, 231, 230, 219,
                ],
            ),
            (
                32usize,
                [
                    243, 143, 159, 255, 38, 103, 151, 108, 246, 85, 35, 230, 43, 68,
                    34, 13, 224, 28, 197, 173, 187, 189, 137, 42, 66, 253, 46, 113,
                    6, 95, 222, 58,
                ],
            ),
            (
                0usize,
                [
                    169, 62, 71, 162, 109, 95, 87, 15, 82, 47, 191, 147, 103, 5,
                    135, 123, 202, 57, 21, 37, 214, 223, 92, 14, 95, 139, 106, 223,
                    31, 190, 31, 147,
                ],
            ),
            (
                1usize,
                [
                    16, 93, 153, 119, 195, 207, 76, 201, 81, 214, 64, 20, 243, 115,
                    127, 49, 93, 97, 42, 44, 137, 193, 215, 23, 74, 250, 96, 218,
                    95, 163, 209, 215,
                ],
            ),
            (
                4usize,
                [
                    42, 32, 73, 21, 194, 100, 136, 30, 76, 68, 119, 187, 67, 210,
                    123, 96, 62, 189, 135, 158, 31, 98, 178, 77, 12, 252, 43, 74,
                    159, 209, 1, 50,
                ],
            ),
            (
                16usize,
                [
                    59, 187, 62, 33, 56, 73, 182, 33, 254, 181, 162, 187, 4, 61,
                    226, 23, 72, 200, 9, 76, 243, 23, 7, 225, 123, 185, 47, 248,
                    34, 179, 4, 182,
                ],
            ),
            (
                24usize,
                [
                    2, 25, 249, 214, 92, 16, 100, 191, 164, 120, 54, 46, 123, 111,
                    205, 206, 245, 162, 153, 63, 93, 68, 39, 122, 116, 48, 149, 196,
                    252, 231, 230, 219,
                ],
            ),
            (
                32usize,
                [
                    243, 143, 159, 255, 38, 103, 151, 108, 246, 85, 35, 230, 43, 68,
                    34, 13, 224, 28, 197, 173, 187, 189, 137, 42, 66, 253, 46, 113,
                    6, 95, 222, 58,
                ],
            ),
        ];
        for (depth, expected) in vectors {
            assert_eq!(roots[depth], expected, "depth {depth}");
        }
    }

    #[test]
    fn endian_tamper_changes_hash() {
        let s = slot(0x0102, 0x0304, 0x0506);
        let mut tampered = s;
        tampered.value = Block128::from(0x0201u128);
        assert_ne!(slot_leaf_hash(s), slot_leaf_hash(tampered));
        assert_ne!(Block128::ZERO, Block128::ONE);
    }
}
