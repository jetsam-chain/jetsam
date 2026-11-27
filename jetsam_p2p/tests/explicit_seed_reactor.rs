// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Seeds given explicitly (--seed), through the real P2P reactor on loopback.
//!
//! Seen on the test chain (2026-11-19): every node sits on a private LAN,
//! where discovered peers are never dialled, and lists the other nodes with
//! --seed. The bootstrap fanout (two seeds, one once a neighbour is stable)
//! left each node with a single connection to lose.

use std::sync::Arc;
use std::time::Duration;

use jetsam_chain::storage::MdbxChainContext;
use jetsam_mempool::{AsyncMempool, ChainView, MempoolConfig};
use jetsam_p2p::{
    BackgroundCapacity, NetworkEvent, NetworkEventReceiver, NetworkEventRecvError, NetworkTopics,
    P2PNetwork,
};
use libp2p::Multiaddr;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explicit_seeds_are_all_dialled_and_kept() {
    const SEEDS: usize = 3;
    let directories = [(); SEEDS + 1].map(|()| tempfile::tempdir().unwrap());
    let mut seeds = Vec::new();
    for directory in &directories[..SEEDS] {
        let port = free_port();
        let seed = start(directory.path(), port).await;
        drain(seed.subscribe());
        seeds.push((seed, port));
    }

    let node = start(directories[SEEDS].path(), free_port()).await;
    let mut events = node.subscribe();
    for (_, port) in &seeds {
        node.dial_seed(loopback(*port)).await;
    }
    let mut connected = std::collections::HashSet::new();
    let all = tokio::time::timeout(Duration::from_secs(60), async {
        while connected.len() < SEEDS {
            if let Ok(NetworkEvent::PeerConnected { peer, .. }) = events.recv().await {
                connected.insert(peer);
            }
        }
    })
    .await;
    assert!(
        all.is_ok(),
        "every explicit seed is dialled: {} of {SEEDS} connected",
        connected.len()
    );

    // Bootstrap over, and still no ordinary neighbour on a private network:
    // the fanout would keep two seeds, the explicit ones all stay.
    node.mark_bootstrap_complete().await;
    let lost = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if let Ok(NetworkEvent::PeerDisconnected(peer)) = events.recv().await {
                return peer;
            }
        }
    })
    .await;
    assert!(lost.is_err(), "an explicit seed was released: {lost:?}");
}
