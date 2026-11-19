// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A TCP flow that dies in silence, through the real P2P reactor on loopback.
//!
//! Seen on the test chain (2026-11-19): the router on the LAN exit sometimes
//! forgets the state of established flows. Their packets are then dropped,
//! no RST and no FIN reaches either end, while new flows pass at once. Both
//! nodes kept the dead connection for about 16 minutes (the kernel's
//! tcp_retries2), and the side holding it refused the peer's new one.
//!
//! [`Nat`] stands in for that router: every flow is relayed between two
//! sockets, and [`Nat::forget_flows`] silences the flows open at that moment
//! (their bytes are read and dropped, their sockets stay open), while the
//! flows opened afterwards are relayed as before.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use jetsam_chain::storage::MdbxChainContext;
use jetsam_mempool::{AsyncMempool, ChainView, MempoolConfig};
use jetsam_p2p::{
    BackgroundCapacity, NetworkEvent, NetworkEventReceiver, NetworkEventRecvError, NetworkTopics,
    P2PNetwork,
};
use libp2p::{Multiaddr, PeerId};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
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

async fn start(
    directory: &std::path::Path,
    port: u16,
) -> (P2PNetwork, tokio::task::JoinHandle<anyhow::Result<()>>) {
    let context = MdbxChainContext::open_or_create(directory).unwrap();
    let view = ChainView::from_mdbx(&context);
    let chain = Arc::new(RwLock::new(context));
    let mempool = AsyncMempool::new(view, MempoolConfig::default());
    P2PNetwork::start(
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
    .unwrap()
}

/// Keeps a reactor whose events nobody reads from blocking on them.
fn drain(mut events: NetworkEventReceiver) {
    tokio::spawn(async move {
        while !matches!(events.recv().await, Err(NetworkEventRecvError::Closed)) {}
    });
}

/// The next peer to connect (`expected`, or any), within `within`.
async fn connected(
    events: &mut NetworkEventReceiver,
    expected: Option<PeerId>,
    within: Duration,
) -> Option<PeerId> {
    tokio::time::timeout(within, async {
        loop {
            match events.recv().await {
                Ok(NetworkEvent::PeerConnected { peer, .. })
                    if expected.is_none_or(|expected| expected == peer) =>
                {
                    return peer
                }
                Err(NetworkEventRecvError::Closed) => std::future::pending::<()>().await,
                _ => {}
            }
        }
    })
    .await
    .ok()
}

/// True once `peer` disconnects within `within`.
async fn disconnected(events: &mut NetworkEventReceiver, peer: PeerId, within: Duration) -> bool {
    tokio::time::timeout(within, async {
        loop {
            match events.recv().await {
                Ok(NetworkEvent::PeerDisconnected(seen)) if seen == peer => return,
                Err(NetworkEventRecvError::Closed) => std::future::pending::<()>().await,
                _ => {}
            }
        }
    })
    .await
    .is_ok()
}

/// The router on the LAN exit, in front of one listening node.
struct Nat {
    port: u16,
    flows: Arc<AtomicU64>,
    forgotten: Arc<AtomicU64>,
}

impl Nat {
    async fn spawn(target: u16) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let flows = Arc::new(AtomicU64::new(0));
        let forgotten = Arc::new(AtomicU64::new(0));
        let (opened, silenced) = (flows.clone(), forgotten.clone());
        tokio::spawn(async move {
            while let Ok((inside, _)) = listener.accept().await {
                let flow = opened.fetch_add(1, Ordering::SeqCst) + 1;
                let silenced = silenced.clone();
                tokio::spawn(async move {
                    let Ok(outside) = tokio::net::TcpStream::connect(("127.0.0.1", target)).await
                    else {
                        return;
                    };
                    let _ = (inside.set_nodelay(true), outside.set_nodelay(true));
                    let (inside_read, inside_write) = inside.into_split();
                    let (outside_read, outside_write) = outside.into_split();
                    tokio::join!(
                        relay(inside_read, outside_write, flow, silenced.clone()),
                        relay(outside_read, inside_write, flow, silenced),
                    );
                });
            }
        });
        Self {
            port,
            flows,
            forgotten,
        }
    }

    /// Every flow open now dies in silence; later flows pass.
    fn forget_flows(&self) {
        self.forgotten
            .store(self.flows.load(Ordering::SeqCst), Ordering::SeqCst);
    }
}

async fn relay(mut from: OwnedReadHalf, mut to: OwnedWriteHalf, flow: u64, forgotten: Arc<AtomicU64>) {
    let mut buffer = vec![0u8; 64 * 1024];
    loop {
        let read = from.read(&mut buffer).await;
        if flow <= forgotten.load(Ordering::SeqCst) {
            if matches!(read, Ok(0) | Err(_)) {
                // Neither a FIN nor an RST crosses a forgotten flow: both
                // halves stay open for as long as the test runs.
                let _hold = (from, to);
                std::future::pending::<()>().await;
            }
            continue;
        }
        match read {
            Ok(0) | Err(_) => {
                let _ = to.shutdown().await;
                return;
            }
            Ok(read) => {
                if to.write_all(&buffer[..read]).await.is_err() {
                    return;
                }
            }
        }
    }
}

/// A node whose connection to its seed dies in silence closes it once its
/// pings go unanswered, then dials the seed again through a new flow.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_connection_that_dies_in_silence_is_closed_and_the_seed_dialled_again() {
    let directories = [(); 2].map(|()| tempfile::tempdir().unwrap());
    let seed_port = free_port();
    let (seed, _seed_task) = start(directories[0].path(), seed_port).await;
    drain(seed.subscribe());
    let nat = Nat::spawn(seed_port).await;

    let (node, _node_task) = start(directories[1].path(), free_port()).await;
    let mut events = node.subscribe();
    node.dial(loopback(nat.port)).await;
    let seed_peer = connected(&mut events, None, Duration::from_secs(120))
        .await
        .expect("the node reaches its seed through the router");

    nat.forget_flows();
    // Two unanswered pings (libp2p-ping reports the second one): at most
    // 2 x (15 s interval + 10 s timeout) after the flow died.
    assert!(
        disconnected(&mut events, seed_peer, Duration::from_secs(60)).await,
        "the dead connection is closed once its pings go unanswered"
    );
    assert!(
        connected(&mut events, Some(seed_peer), Duration::from_secs(90))
            .await
            .is_some(),
        "the seed is dialled again through a new flow"
    );
}
