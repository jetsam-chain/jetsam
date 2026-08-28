// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Deterministic differential fuzzing of the final 16-leaf statement.

use jetsam_core::Block128;
use jetsam_gkr::oracle::evaluate_spine;
use jetsam_gkr::{SpineCircuit, SpineInputs};
use jetsam_poseidon2b::primitives::hash_tx8x2_leaves;

#[test]
fn deterministic_raw_leaf_differential() {
    let circuit = SpineCircuit::build();
    let mut state = 0x8f31_4d29_53a1_c7e5u128;
    for case in 0..64 {
        let leaves = std::array::from_fn(|leaf| {
            std::array::from_fn(|lane| {
                state = state
                    .wrapping_mul(0xda94_2042_e4dd_58b5)
                    .wrapping_add((case * 32 + leaf * 2 + lane + 1) as u128);
                Block128::from(state)
            })
        });
        let inputs = SpineInputs { leaves };
        assert_eq!(
            evaluate_spine(&circuit, &inputs).tx_body_hash_bytes(),
            hash_tx8x2_leaves(&leaves).0,
            "native/GKR drift in deterministic case {case}"
        );
    }
}
