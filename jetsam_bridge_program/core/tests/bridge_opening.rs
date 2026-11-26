// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The program's bridge opening `F` (V4-d) on each of its clocks: the test chain's opens 544 blocks after its J
//! (D104: 8 160 + 544 = 8 704), on the program's own `testnet` too, whose guest build does not hold
//! `jetsam_vault_rule/testnet`; the public network keeps none (no J yet) and 1 440 blocks after a J (D44); the
//! rehearsal J_R + 32 = 128. `jetsam_bridge_xtest` checks the same values against the node's.

use jetsam_bridge_program::deposit::{DEPOSIT_RULE, V1_7_ACTIVATION_HEIGHT};

#[test]
fn the_programs_bridge_opening_of_each_clock() {
    assert_eq!(DEPOSIT_RULE.activation_height, V1_7_ACTIVATION_HEIGHT);
    #[cfg(all(feature = "testnet", not(any(feature = "rehearsal-v1-7", feature = "test-clock-j69120"))))]
    {
        assert_eq!(V1_7_ACTIVATION_HEIGHT, Some(8_160));
        assert_eq!(DEPOSIT_RULE.deposits_from_height, Some(8_704));
        assert!(!DEPOSIT_RULE.deposits_open_at(8_704));
        assert!(DEPOSIT_RULE.deposits_open_at(8_705));
    }
    #[cfg(feature = "rehearsal-v1-7")]
    assert_eq!(DEPOSIT_RULE.deposits_from_height, Some(128));
    #[cfg(all(feature = "test-clock-j69120", not(feature = "rehearsal-v1-7")))]
    assert_eq!(DEPOSIT_RULE.deposits_from_height, Some(69_120 + 1_440));
    #[cfg(not(any(feature = "testnet", feature = "rehearsal-v1-7", feature = "test-clock-j69120")))]
    {
        if jetsam_vault_rule::TESTNET_PROFILE {
            // A build holding the node's `jetsam_chain/testnet`: the test chain's clock (M5).
            assert_eq!(DEPOSIT_RULE.deposits_from_height, Some(8_704));
        } else {
            // The public network: J = 61 440 (26/11/2026), F = J + 1 440 = 62 880.
            assert_eq!(V1_7_ACTIVATION_HEIGHT, Some(61_440));
            assert_eq!(DEPOSIT_RULE.deposits_from_height, Some(62_880));
            assert!(!DEPOSIT_RULE.deposits_open_at(62_880));
            assert!(DEPOSIT_RULE.deposits_open_at(62_881));
            assert_eq!(jetsam_vault_rule::bridge_deposits_from_height(Some(69_120)), Some(70_560));
        }
    }
}
