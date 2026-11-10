// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! D123: the program proves the deposits the nodes accept under the caps of each block's height. The test chain's
//! program (its own `testnet`, whose guest build does not hold `jetsam_vault_rule/testnet`, or a build holding the
//! node's `jetsam_chain/testnet`) keeps 1 000 JTM a deposit below the test chain's caps height
//! (`jetsam_vault_rule::TESTNET_CAPS_HEIGHT`, not chosen: every height); the public network, the rehearsal and the
//! test clock of the G6 vectors have D123's 50 000 JTM from their J. `jetsam_bridge_xtest` checks the same against
//! the node's rule.

use jetsam_bridge_program::deposit::DEPOSIT_RULE;

const JTM: u64 = 1_000_000;

#[test]
fn the_programs_largest_deposit_of_each_clock() {
    let test_chain = cfg!(all(feature = "testnet", not(any(feature = "rehearsal-v1-7", feature = "test-clock-j69120"))))
        || (cfg!(not(any(feature = "testnet", feature = "rehearsal-v1-7", feature = "test-clock-j69120")))
            && jetsam_vault_rule::TESTNET_PROFILE);
    assert_eq!(DEPOSIT_RULE.max_deposit, 50_000 * JTM);
    if test_chain {
        assert_eq!(DEPOSIT_RULE.legacy, Some(jetsam_vault_rule::TESTNET_LEGACY_DEPOSIT_CAP));
        assert_eq!(jetsam_vault_rule::TESTNET_CAPS_HEIGHT, Some(8_910));
        for h in [8_705, 8_909] {
            assert_eq!(DEPOSIT_RULE.max_deposit_at(h), 1_000 * JTM, "{h}");
        }
        for h in [8_910, 1 << 40] {
            assert_eq!(DEPOSIT_RULE.max_deposit_at(h), 50_000 * JTM, "{h}");
        }
    } else {
        assert_eq!(DEPOSIT_RULE.legacy, None);
        assert_eq!(DEPOSIT_RULE.max_deposit_at(1 << 40), 50_000 * JTM);
    }
}
