// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The software version a node announces through Identify, read back through
//! the real P2P reactor (`P2PNetwork::start`) on loopback: what an adoption
//! count before a hardfork is built on. Then a peer announcing a hostile
//! version (control characters, non-ASCII, 2 000 bytes): it is admitted
//! exactly as any compatible peer is, and only a bounded, printable copy of
//! its claim is kept.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use jetsam_chain::storage::MdbxChainContext;
use jetsam_mempool::{AsyncMempool, ChainView, MempoolConfig};
use jetsam_p2p::behaviour::{NodeBehaviour, NodeBehaviourEvent};
use jetsam_p2p::network::NetworkEvent;
use jetsam_p2p::network_profile::{NetworkProfile, NetworkProfileResponse};
use jetsam_p2p::peer_versions::{
    local_agent_version, ObservedPeerVersion, MAX_AGENT_VERSION_CHARS,
};
use jetsam_p2p::{BackgroundCapacity, NetworkTopics, P2PNetwork};
use libp2p::{
    identify, noise, request_response, swarm::SwarmEvent, tcp, yamux, Multiaddr, PeerId,
    SwarmBuilder,
};
use tokio::sync::RwLock;

const PROOF_BANK: [u8; 32] = [0x7A; 32];

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn topics() -> NetworkTopics {
    NetworkTopics::for_network_cfg(&jetsam_chain::consensus::NetworkConfig::mainnet())
}

async fn start(directory: &std::path::Path, port: u16) -> P2PNetwork {
    let context = MdbxChainContext::open_or_create(directory).unwrap();
    let view = ChainView::from_mdbx(&context);
    let chain = Arc::new(RwLock::new(context));
    let mempool = AsyncMempool::new(view, MempoolConfig::default());
    let listen: Multiaddr = format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap();
    let (network, _task) = P2PNetwork::start(
        vec![listen],
        Vec::new(),
        chain,
        mempool,
        topics(),
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

async fn connected_peer(
    events: &mut jetsam_p2p::network::NetworkEventReceiver,
    expected: Option<PeerId>,
) -> PeerId {
    tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            if let Ok(NetworkEvent::PeerConnected { peer, .. }) = events.recv().await {
                if expected.is_none_or(|expected| expected == peer) {
                    return peer;
                }
            }
        }
    })
    .await
    .expect("the peer is admitted")
}

/// What `node` reports about `peer` once that peer is connected and its
/// network profile verified.
async fn verified_entry(node: &P2PNetwork, peer: PeerId) -> ObservedPeerVersion {
    tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            let observed = P2PNetwork::peer_versions_via(&node.cmd_tx)
                .await
                .expect("the reactor answers");
            if let Some(entry) = observed.into_iter().find(|entry| {
                entry.peer_id == peer.to_string() && entry.connected && entry.profile_verified
            }) {
                return entry;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    })
    .await
    .expect("the connected peer's version is reported")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_reactors_report_the_release_each_other_announces() {
    let (dialer_dir, listener_dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let listener_port = free_port();
    let listener = start(listener_dir.path(), listener_port).await;
    let dialer = start(dialer_dir.path(), free_port()).await;
    let mut dialer_events = dialer.subscribe();
    let mut listener_events = listener.subscribe();
    dialer
        .dial(
            format!("/ip4/127.0.0.1/tcp/{listener_port}")
                .parse()
                .unwrap(),
        )
        .await;
    let listener_peer = connected_peer(&mut dialer_events, None).await;
    let dialer_peer = connected_peer(&mut listener_events, None).await;

    let announced = local_agent_version();
    assert!(announced.starts_with(concat!("jetsam/", env!("CARGO_PKG_VERSION"), " (v1.6=")));
    // Outbound: what a node that dials a seed sees.
    let seen_by_dialer = verified_entry(&dialer, listener_peer).await;
    assert_eq!(seen_by_dialer.agent_version, announced);
    // Inbound: what a seed sees of the nodes that dial it, which is where an
    // adoption count is read.
    let seen_by_listener = verified_entry(&listener, dialer_peer).await;
    assert_eq!(seen_by_listener.agent_version, announced);
    assert!(seen_by_listener.first_seen_unix <= seen_by_listener.last_seen_unix);
}

/// A peer with the production behaviour and a hostile version string. It
/// answers the network-profile handshake honestly and nothing else.
async fn spawn_hostile_peer(agent: String) -> (PeerId, u16) {
    let protocol_id = topics().protocol_id;
    let mut swarm = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )
        .unwrap()
        .with_dns()
        .unwrap()
        .with_relay_client(noise::Config::new, yamux::Config::default)
        .unwrap()
        .with_behaviour(move |key, relay_client| {
            let mut behaviour = NodeBehaviour::new(
                key,
                &protocol_id,
                relay_client,
                BackgroundCapacity::Full,
                false,
                false,
                false,
                false,
            )?;
            behaviour.identify = identify::Behaviour::new(
                identify::Config::new("/jetsam/1.0.0".into(), key.public())
                    .with_agent_version(agent),
            );
            Ok(behaviour)
        })
        .unwrap()
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(300)))
        .build();
    let port = free_port();
    swarm
        .listen_on(format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap())
        .unwrap();
    let peer = *swarm.local_peer_id();
    tokio::spawn(async move {
        loop {
            if let SwarmEvent::Behaviour(NodeBehaviourEvent::NetworkProfileSync(
                request_response::Event::Message {
                    message: request_response::Message::Request { channel, .. },
                    ..
                },
            )) = swarm.select_next_some().await
            {
                let _ = swarm.behaviour_mut().network_profile_sync.send_response(
                    channel,
                    NetworkProfileResponse {
                        profile: NetworkProfile::for_proof_bank(PROOF_BANK),
                    },
                );
            }
        }
    });
    (peer, port)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_hostile_version_is_bounded_and_changes_nothing_about_admission() {
    // Control characters, an ANSI escape, non-ASCII, and far more than any
    // honest version needs; still under Identify's own 4 KiB message cap.
    let hostile = format!(
        "jetsam/1.6.0 (v1.6=5760)\u{1b}[2J\r\n\0é漢{}",
        "9".repeat(2_000)
    );
    let (hostile_peer, hostile_port) = spawn_hostile_peer(hostile).await;
    let node_dir = tempfile::tempdir().unwrap();
    let node = start(node_dir.path(), free_port()).await;
    let mut events = node.subscribe();
    node.dial(
        format!("/ip4/127.0.0.1/tcp/{hostile_port}")
            .parse()
            .unwrap(),
    )
    .await;

    // Admitted like any compatible peer: the version is not part of the
    // decision, so a lie about it buys nothing and costs nothing.
    connected_peer(&mut events, Some(hostile_peer)).await;
    let entry = verified_entry(&node, hostile_peer).await;
    assert_eq!(entry.agent_version.len(), MAX_AGENT_VERSION_CHARS);
    assert!(entry
        .agent_version
        .bytes()
        .all(|byte| (0x20..=0x7e).contains(&byte)));
    assert!(entry
        .agent_version
        .starts_with("jetsam/1.6.0 (v1.6=5760)?[2J"));
    assert_eq!(
        P2PNetwork::peer_count_via(&node.cmd_tx).await,
        Some(1),
        "the hostile peer is an ordinary dispatchable peer"
    );
}
