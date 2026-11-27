// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! T7.1: destination and guardian intents through the real P2P reactor
//! (`P2PNetwork::start`, the node's `NodeBehaviour`) on loopback. Three
//! nodes in a line, A — B — C: an intent A publishes reaches B, and goes on
//! to C only once B accepts it; one B refuses, or junk, goes no further; a
//! node without the bridge topic never sees one.

use std::sync::Arc;
use std::time::Duration;

use jetsam_chain::storage::MdbxChainContext;
use jetsam_mempool::{AsyncMempool, ChainView, MempoolConfig};
use jetsam_p2p::bridge_intent_gossip::BridgeIntentFamily;
use jetsam_p2p::network::NetworkEvent;
use jetsam_p2p::{NetworkCommand, NetworkTopics, P2PNetwork};
use libp2p::{gossipsub, Multiaddr, PeerId};
use tokio::sync::RwLock;

mod support;
use support::bridge_intents::{destination_intent_bytes, guardian_intent_bytes};

const TOPIC: &str = "/jetsam/test-bridge/gossip/bridge-intents/1";

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn start(directory: &std::path::Path, port: u16, bridge: bool) -> P2PNetwork {
    let context = MdbxChainContext::open_or_create(directory).unwrap();
    let view = ChainView::from_mdbx(&context);
    let chain = Arc::new(RwLock::new(context));
    let mempool = AsyncMempool::new(view, MempoolConfig::default());
    let mut topics =
        NetworkTopics::for_network_cfg(&jetsam_chain::consensus::NetworkConfig::mainnet());
    topics.bridge_intents = bridge.then(|| TOPIC.to_string());
    let listen: Multiaddr = format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap();
    let (network, _task) = P2PNetwork::start(
        vec![listen],
        Vec::new(),
        chain,
        mempool,
        topics,
        [0x7B; 32],
        directory.to_path_buf(),
        jetsam_p2p::BackgroundCapacity::Full,
        false,
        false,
        None,
    )
    .unwrap();
    network
}

/// A relayed intent: who handed it in, its family, its bytes, its gossip id.
type Relayed = (PeerId, BridgeIntentFamily, Vec<u8>, gossipsub::MessageId);

async fn next_relayed(
    events: &mut jetsam_p2p::network::NetworkEventReceiver,
    timeout: Duration,
    wanted: &[u8],
) -> Option<Relayed> {
    tokio::time::timeout(timeout, async {
        loop {
            if let Ok(NetworkEvent::BridgeIntentRelayed {
                from,
                family,
                intent_bytes,
                gossip_message_id,
            }) = events.recv().await
            {
                if intent_bytes.as_ref() == wanted {
                    return (from, family, intent_bytes.to_vec(), gossip_message_id);
                }
            }
        }
    })
    .await
    .ok()
}

/// Any relayed intent at all within `timeout`.
async fn any_relayed(
    events: &mut jetsam_p2p::network::NetworkEventReceiver,
    timeout: Duration,
) -> Option<Vec<u8>> {
    tokio::time::timeout(timeout, async {
        loop {
            if let Ok(NetworkEvent::BridgeIntentRelayed { intent_bytes, .. }) = events.recv().await
            {
                return intent_bytes.to_vec();
            }
        }
    })
    .await
    .ok()
}

/// `from` dials the node listening on `to_port`; the next peer `events`
/// (the dialler's) sees connected.
async fn connect(
    from: &P2PNetwork,
    events: &mut jetsam_p2p::network::NetworkEventReceiver,
    to_port: u16,
) -> PeerId {
    from.dial(format!("/ip4/127.0.0.1/tcp/{to_port}").parse().unwrap())
        .await;
    next_connected(events).await
}

async fn next_connected(events: &mut jetsam_p2p::network::NetworkEventReceiver) -> PeerId {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Ok(NetworkEvent::PeerConnected { peer, .. }) = events.recv().await {
                return peer;
            }
        }
    })
    .await
    .expect("the two nodes connect")
}

/// A publishes `make(attempt)` until B relays one to its node (the mesh
/// forms within a few heartbeats); returns the bytes B saw and B's event.
async fn publish_until_b_sees(
    a: &P2PNetwork,
    b_events: &mut jetsam_p2p::network::NetworkEventReceiver,
    make: impl Fn(u64) -> Vec<u8>,
) -> Relayed {
    for attempt in 0..40u64 {
        let bytes = make(attempt);
        a.cmd_tx
            .send(NetworkCommand::BroadcastBridgeIntent {
                intent_bytes: Arc::from(bytes.clone()),
            })
            .await
            .unwrap();
        if let Some(relayed) = next_relayed(b_events, Duration::from_secs(2), &bytes).await {
            return relayed;
        }
    }
    panic!("B never received an intent A published");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_intent_crosses_a_node_only_once_that_node_accepts_it() {
    let (dir_a, dir_b, dir_c, dir_d) = (
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
        tempfile::tempdir().unwrap(),
    );
    let (port_a, port_c) = (free_port(), free_port());
    let a = start(dir_a.path(), port_a, true).await;
    let b = start(dir_b.path(), free_port(), true).await;
    let c = start(dir_c.path(), port_c, true).await;
    // D joins no bridge topic (a node whose v1.7 height is not armed).
    let d = start(dir_d.path(), free_port(), false).await;
    let mut b_events = b.subscribe();
    let mut c_events = c.subscribe();
    let mut d_events = d.subscribe();
    let mut a_events = a.subscribe();
    let a_peer = connect(&b, &mut b_events, port_a).await;
    connect(&b, &mut b_events, port_c).await;
    let b_peer_seen_by_c = next_connected(&mut c_events).await;
    connect(&d, &mut d_events, port_a).await;
    // A sees both of its peers before it publishes.
    next_connected(&mut a_events).await;
    next_connected(&mut a_events).await;

    // 1. A guardian intent: B's node gets it from A, exactly; C nothing
    // until B's node resolves it.
    let (from, family, bytes, id) =
        publish_until_b_sees(&a, &mut b_events, guardian_intent_bytes).await;
    assert_eq!(from, a_peer);
    assert_eq!(family, BridgeIntentFamily::Guardian);
    assert_eq!(
        next_relayed(&mut c_events, Duration::from_secs(3), &bytes).await,
        None,
        "nothing goes past B before its node accepts it"
    );
    b.cmd_tx
        .send(NetworkCommand::ResolveTxGossip {
            message_id: id,
            propagation_source: from,
            acceptance: gossipsub::MessageAcceptance::Accept,
        })
        .await
        .unwrap();
    let (c_from, c_family, c_bytes, _) =
        next_relayed(&mut c_events, Duration::from_secs(20), &bytes)
            .await
            .expect("accepted by B, it reaches C");
    assert_eq!(c_from, b_peer_seen_by_c);
    assert_eq!(c_family, BridgeIntentFamily::Guardian);
    assert_eq!(c_bytes, bytes);

    // 2. A destination intent B's node refuses: never at C.
    let (from, family, bytes, id) =
        publish_until_b_sees(&a, &mut b_events, |n| destination_intent_bytes(100 + n)).await;
    assert_eq!(family, BridgeIntentFamily::Destination);
    b.cmd_tx
        .send(NetworkCommand::ResolveTxGossip {
            message_id: id,
            propagation_source: from,
            acceptance: gossipsub::MessageAcceptance::Reject,
        })
        .await
        .unwrap();
    assert_eq!(
        next_relayed(&mut c_events, Duration::from_secs(5), &bytes).await,
        None,
        "refused by B, it goes no further"
    );

    // 3. Junk on the topic never reaches a node.
    for attempt in 0..5u8 {
        a.cmd_tx
            .send(NetworkCommand::BroadcastBridgeIntent {
                intent_bytes: Arc::from(vec![0x4A, attempt, 1, 2, 3]),
            })
            .await
            .unwrap();
    }
    if let Some(seen) = any_relayed(&mut b_events, Duration::from_secs(5)).await {
        panic!("junk handed to the node: {seen:?}");
    }

    // 4. D, without the topic, saw none of it.
    assert_eq!(
        any_relayed(&mut d_events, Duration::from_secs(1)).await,
        None
    );
}
