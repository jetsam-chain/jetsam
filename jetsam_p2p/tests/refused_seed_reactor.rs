// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A seed proven on a refused branch, through the real P2P reactor on loopback.

use std::sync::Arc;
use std::time::Duration;

use jetsam_chain::storage::MdbxChainContext;
use jetsam_mempool::{AsyncMempool, ChainView, MempoolConfig};
use jetsam_p2p::{
    BackgroundCapacity, NetworkCommand, NetworkEvent, NetworkEventReceiver, NetworkEventRecvError,
    NetworkTopics, P2PNetwork,
};
use libp2p::{Multiaddr, PeerId};
use tokio::sync::RwLock;

const PROOF_BANK: [u8; 32] = [0x7A; 32];

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn loopback(port: u16) -> Multiaddr {
    format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap()
}

async fn start(directory: &std::path::Path, port: u16) -> P2PNetwork {
    let context = MdbxChainContext::open_or_create(directory).unwrap();
    let view = ChainView::from_mdbx(&context);
    let chain = Arc::new(RwLock::new(context));
    let mempool = AsyncMempool::new(view, MempoolConfig::default());
    let (network, _task) = P2PNetwork::start(
        vec![loopback(port)],
        Vec::new(),
        chain,
        mempool,
        NetworkTopics::for_network_cfg(&jetsam_chain::consensus::NetworkConfig::mainnet()),
        PROOF_BANK,
        directory.to_path_buf(),
        BackgroundCapacity::Full,
        false,
        false,
        None,
    )
    .unwrap();
    network
}

/// Keeps a reactor whose events nobody reads from blocking on them.
fn drain(mut events: NetworkEventReceiver) {
    tokio::spawn(async move {
        while !matches!(events.recv().await, Err(NetworkEventRecvError::Closed)) {}
    });
}

/// True once `peer` connects (`connected`) or disconnects, false if it does not within `within`.
async fn peer_event(
    events: &mut NetworkEventReceiver,
    peer: PeerId,
    connected: bool,
    within: Duration,
) -> bool {
    tokio::time::timeout(within, async {
        loop {
            match events.recv().await {
                Ok(NetworkEvent::PeerConnected { peer: seen, .. }) if connected && seen == peer => {
                    return
                }
                Ok(NetworkEvent::PeerDisconnected(seen)) if !connected && seen == peer => return,
                Err(NetworkEventRecvError::Closed) => std::future::pending::<()>().await,
                _ => {}
            }
        }
    })
    .await
    .is_ok()
}

/// Seen on the 1.6.1 private test chain (2026-10-25): a Kademlia lookup dialled the released seed 28 s later.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_seed_released_on_a_refused_branch_is_not_dialled_again_by_discovery() {
    let directories = [(); 3].map(|()| tempfile::tempdir().unwrap());
    let (refused_port, healthy_port) = (free_port(), free_port());
    let refused_seed = start(directories[0].path(), refused_port).await;
    drain(refused_seed.subscribe());
    let healthy_seed = start(directories[1].path(), healthy_port).await;
    let mut healthy_events = healthy_seed.subscribe();
    // The healthy seed knows the refused one, as any peer of the network may.
    healthy_seed.dial(loopback(refused_port)).await;
    let refused_peer = tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            if let Ok(NetworkEvent::PeerConnected { peer, .. }) = healthy_events.recv().await {
                return peer;
            }
        }
    })
    .await
    .expect("the healthy seed reaches the refused one");
    drain(healthy_events);

    let node = start(directories[2].path(), free_port()).await;
    let mut events = node.subscribe();
    node.dial(loopback(refused_port)).await;
    node.dial(loopback(healthy_port)).await;
    let mut connected = std::collections::HashSet::new();
    tokio::time::timeout(Duration::from_secs(120), async {
        while connected.len() < 2 || !connected.contains(&refused_peer) {
            if let Ok(NetworkEvent::PeerConnected { peer, .. }) = events.recv().await {
                connected.insert(peer);
            }
        }
    })
    .await
    .expect("both seeds connect");

    node.cmd_tx
        .try_send(NetworkCommand::RefusedBranchPeers {
            peers: vec![refused_peer],
        })
        .unwrap();
    assert!(
        peer_event(&mut events, refused_peer, false, Duration::from_secs(30)).await,
        "the seed's connection is released"
    );
    assert!(
        !peer_event(&mut events, refused_peer, true, Duration::from_secs(45)).await,
        "the released seed is dialled again while the verdict holds"
    );
}
