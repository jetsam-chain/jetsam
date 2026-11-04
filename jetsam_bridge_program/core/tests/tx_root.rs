// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The program's `tx_root` of a whole block skips the empty subtrees of the 256-leaf tree: it must be the
//! full tree's root (`root_and_path`, the v2 port checked against the node) for every leaf count.

use jetsam_bridge_program::tx;

#[test]
fn the_sparse_root_is_the_full_trees_root_for_every_count() {
    let mut seed = 0x7478_5f72_6f6f_7433u64;
    let mut leaf = || {
        let mut l = [0u8; 32];
        for b in l.iter_mut() {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            *b = (seed >> 56) as u8;
        }
        l
    };
    let leaves: Vec<[u8; 32]> = (0..tx::TX_TREE_LEAVES).map(|_| leaf()).collect();
    for n in 0..=tx::TX_TREE_LEAVES {
        assert_eq!(tx::root(&leaves[..n]), tx::root_and_path(&leaves[..n], 0).0, "{n} leaves");
    }
    // a zero leaf is a leaf: the count, not the content, ends the tree
    let zeros = [[0u8; 32]; 3];
    for n in 0..=3 {
        assert_eq!(tx::root(&zeros[..n]), tx::root_and_path(&zeros[..n], 0).0, "{n} zero leaves");
    }
    assert_ne!(tx::root(&zeros[..2]), tx::root(&zeros[..3]));
}
