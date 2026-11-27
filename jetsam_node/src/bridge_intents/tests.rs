// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use jetsam_chain::storage::MdbxChainContext;
use jetsam_mempool::{ChainView, MempoolConfig};

#[path = "../../../jetsam_p2p/tests/support/bridge_intents.rs"]
mod support;

fn mempool() -> (tempfile::TempDir, AsyncMempool) {
    let directory = tempfile::tempdir().unwrap();
    let context = MdbxChainContext::open_or_create(directory.path()).unwrap();
    let mempool = AsyncMempool::new(ChainView::from_mdbx(&context), MempoolConfig::default());
    (directory, mempool)
}

/// A1 m13 through the relay (A3-5's P2P side): the guardian rate limit
/// counts each relaying peer apart — a peer flooding signed objects is
/// refused before any signature check, another peer is not.
#[tokio::test]
async fn the_guardian_rate_limit_holds_per_relaying_peer() {
    let (_directory, pool) = mempool();
    let (flooder, other) = (PeerId::random(), PeerId::random());
    let limit = jetsam_mempool::guardian::GUARDIAN_RATE_LIMIT_PER_SECOND;
    for fill in 0..limit as u8 {
        let result = submit_relayed_bridge_intent(
            &pool,
            flooder,
            BridgeIntentFamily::Guardian,
            &support::attestation_intent_bytes(fill),
        )
        .await;
        assert!(
            !matches!(result, Err(SubmitError::GuardianRateLimited { .. })),
            "{fill}: within the limit ({result:?})"
        );
    }
    let flooded = submit_relayed_bridge_intent(
        &pool,
        flooder,
        BridgeIntentFamily::Guardian,
        &support::attestation_intent_bytes(0xF0),
    )
    .await;
    assert_eq!(
        flooded.as_ref().err().map(ToString::to_string),
        Some(
            SubmitError::GuardianRateLimited {
                origin: relay_source(&flooder)
            }
            .to_string()
        )
    );
    let fresh = submit_relayed_bridge_intent(
        &pool,
        other,
        BridgeIntentFamily::Guardian,
        &support::attestation_intent_bytes(0xF1),
    )
    .await;
    assert!(
        !matches!(fresh, Err(SubmitError::GuardianRateLimited { .. })),
        "another peer is not limited by the flooder ({fresh:?})"
    );
    assert!(matches!(
        relayed_intent_acceptance(&flooded),
        MessageAcceptance::Ignore
    ));
}

/// The gossip verdict: held → forwarded; not an intent → refused; anything
/// else → dropped without penalty.
#[tokio::test]
async fn a_relayed_intent_is_forwarded_only_once_held() {
    let (_directory, pool) = mempool();
    assert!(matches!(
        relayed_intent_acceptance(&Ok(())),
        MessageAcceptance::Accept
    ));
    let junk = submit_relayed_bridge_intent(
        &pool,
        PeerId::random(),
        BridgeIntentFamily::Destination,
        b"JTMDIN01 not an intent",
    )
    .await;
    assert!(
        matches!(junk, Err(SubmitError::MalformedIntent(_))),
        "{junk:?}"
    );
    assert!(matches!(
        relayed_intent_acceptance(&junk),
        MessageAcceptance::Reject
    ));
    // A well-formed release this node cannot hold (no v1.7 clock here).
    let unheld = submit_relayed_bridge_intent(
        &pool,
        PeerId::random(),
        BridgeIntentFamily::Destination,
        &support::destination_intent_bytes(4),
    )
    .await;
    assert!(unheld.is_err());
    assert!(!matches!(unheld, Err(SubmitError::MalformedIntent(_))));
    assert!(matches!(
        relayed_intent_acceptance(&unheld),
        MessageAcceptance::Ignore
    ));
}
