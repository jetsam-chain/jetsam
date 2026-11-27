// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;

#[path = "../../tests/support/bridge_intents.rs"]
mod support;

#[test]
fn bridge_intent_family_reads_exactly_one_intent_of_either_family() {
    let guardian = support::guardian_intent_bytes(10);
    let destination = support::destination_intent_bytes(4);
    assert_eq!(
        bridge_intent_family(&guardian),
        Some(BridgeIntentFamily::Guardian)
    );
    assert_eq!(
        bridge_intent_family(&destination),
        Some(BridgeIntentFamily::Destination)
    );
    for bytes in [&guardian, &destination] {
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(bridge_intent_family(&trailing), None, "a byte after it");
        assert_eq!(
            bridge_intent_family(&bytes[..bytes.len() - 1]),
            None,
            "truncated"
        );
        let mut magic = bytes.clone();
        magic[7] ^= 1;
        assert_eq!(bridge_intent_family(&magic), None, "another magic");
    }
    // A transaction is not an intent, nor is nothing.
    let plain = support::payment(Address([1; 32]), &[], 50_000)
        .to_bytes()
        .unwrap();
    assert_eq!(bridge_intent_family(&plain), None);
    assert_eq!(bridge_intent_family(&[]), None);
    // Over the bound: refused before any decoding.
    let mut huge = jetsam_mempool::guardian::GUARDIAN_INTENT_MAGIC.to_vec();
    huge.resize(MAX_BRIDGE_INTENT_BYTES + 1, 0);
    assert_eq!(bridge_intent_family(&huge), None);
}

use jetsam_poseidon2b::primitives::Address;

#[test]
fn the_transmit_limit_grows_only_with_the_bridge_topic() {
    let base = jetsam_chain::consensus::wire_limits::MAX_TX_INTENT_BYTES_GLOBAL;
    assert_eq!(gossip_max_transmit_size(base, false), base);
    assert_eq!(
        gossip_max_transmit_size(base, true),
        MAX_BRIDGE_INTENT_BYTES.max(base)
    );
    assert!(
        MAX_BRIDGE_INTENT_BYTES > base,
        "an intent carries a transaction and more"
    );
    assert!(MAX_BRIDGE_INTENT_BYTES >= jetsam_mempool::guardian::MAX_GUARDIAN_INTENT_BYTES);
    assert!(MAX_BRIDGE_INTENT_BYTES >= jetsam_mempool::destination::MAX_DESTINATION_INTENT_BYTES);
}

#[test]
fn a_relaying_peer_is_rate_limited_and_the_topic_byte_budget_holds() {
    let mut gossip = BridgeIntentGossip::new(Some("/test/bridge"));
    let start = Instant::now();
    let alice = PeerId::random();
    let bob = PeerId::random();
    let intent = support::guardian_intent_bytes(10);
    for _ in 0..BRIDGE_INTENT_RATE_MAX {
        assert_eq!(
            gossip.judge(alice, &intent, start),
            BridgeIntentVerdict::ToNode(BridgeIntentFamily::Guardian)
        );
    }
    assert_eq!(
        gossip.judge(alice, &intent, start),
        BridgeIntentVerdict::Ignore
    );
    // Junk counts against the rate too, and is refused on its own merits.
    assert_eq!(
        gossip.judge(bob, b"junk", start),
        BridgeIntentVerdict::Reject
    );
    assert_eq!(
        gossip.judge(bob, &intent, start),
        BridgeIntentVerdict::ToNode(BridgeIntentFamily::Guardian)
    );
    // The next window starts afresh.
    let later = start + BRIDGE_INTENT_RATE_WINDOW + Duration::from_millis(1);
    assert_eq!(
        gossip.judge(alice, &intent, later),
        BridgeIntentVerdict::ToNode(BridgeIntentFamily::Guardian)
    );
    // A disconnected peer's count is forgotten.
    gossip.forget_peer(&bob);
    assert!(!gossip.rates.contains_key(&bob));

    // The byte budget of the topic, all peers together.
    let mut gossip = BridgeIntentGossip::new(Some("/test/bridge"));
    let big = vec![0u8; MAX_BRIDGE_INTENT_BYTES];
    let mut passed = 0usize;
    let mut verdicts = Vec::new();
    for _ in 0..(BRIDGE_INTENT_BYTES_PER_WINDOW / MAX_BRIDGE_INTENT_BYTES + 2) {
        let verdict = gossip.judge(PeerId::random(), &big, start);
        if verdict == BridgeIntentVerdict::Reject {
            passed += big.len();
        }
        verdicts.push(verdict);
    }
    assert!(passed <= BRIDGE_INTENT_BYTES_PER_WINDOW);
    assert_eq!(verdicts.last(), Some(&BridgeIntentVerdict::Ignore));
}

#[test]
fn the_peers_whose_rate_is_remembered_are_bounded() {
    let mut gossip = BridgeIntentGossip::new(Some("/test/bridge"));
    let start = Instant::now();
    for _ in 0..MAX_RATE_TRACKED_PEERS {
        gossip.judge(PeerId::random(), b"x", start);
    }
    assert_eq!(gossip.rates.len(), MAX_RATE_TRACKED_PEERS);
    // Still fresh: a new peer is dropped rather than grow the map.
    assert_eq!(
        gossip.judge(PeerId::random(), b"x", start),
        BridgeIntentVerdict::Ignore
    );
    assert_eq!(gossip.rates.len(), MAX_RATE_TRACKED_PEERS);
    // Stale: they make room.
    let later = start + BRIDGE_INTENT_RATE_WINDOW + Duration::from_millis(1);
    assert_eq!(
        gossip.judge(PeerId::random(), b"x", later),
        BridgeIntentVerdict::Reject
    );
    assert_eq!(gossip.rates.len(), 1);
}

#[test]
fn the_bridge_topic_exists_only_on_an_armed_v1_7_clock() {
    let cfg = jetsam_chain::consensus::NetworkConfig::mainnet();
    let topics = crate::protocol::NetworkTopics::for_network_cfg(&cfg);
    match jetsam_chain::consensus::params::V1_7_ACTIVATION_HEIGHT {
        None => assert_eq!(topics.bridge_intents, None),
        Some(_) => assert_eq!(
            topics.bridge_intents.as_deref(),
            Some(format!("{}/gossip/bridge-intents/1", cfg.p2p_protocol_id).as_str())
        ),
    }
    // Every other topic is unchanged by it.
    assert_eq!(
        topics.client_proofs,
        format!("{}/gossip/client-proofs/1", cfg.p2p_protocol_id)
    );
}
