// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The onion modes through the real P2P reactor (`P2PNetwork`), on loopback:
//! a fake ControlPort, a fake SOCKS proxy that forwards to a local peer, and
//! a peer with the production behaviour that records every Identify message
//! it receives. Nothing here reaches Tor or leaves the host.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::StreamExt;
use jetsam_chain::storage::MdbxChainContext;
use jetsam_mempool::{AsyncMempool, ChainView, MempoolConfig};
use libp2p::{
    identify, noise, request_response, swarm::SwarmEvent, tcp, yamux, Multiaddr, PeerId,
    SwarmBuilder,
};
use tokio::sync::RwLock;

use crate::behaviour::{NodeBehaviour, NodeBehaviourEvent};
use crate::network_profile::{NetworkProfile, NetworkProfileResponse};
use crate::onion::config::{OnionConfig, OnionMode, OnionSettings};
use crate::onion::key_store::tests::SYNTHETIC;
use crate::onion::test_support::{FakeSocks, FakeSocksConfig, FakeTor, FakeTorConfig};
use crate::{BackgroundCapacity, NetworkTopics, P2PNetwork};

const PROOF_BANK: [u8; 32] = [0x7A; 32];
/// A published v3 address, handed to the fake proxy as a name only.
const REMOTE: &str = "2gzyxa5ihm7nsggfxnu52rck2vv4rvmdlkiu3zzui5du4xyclen53wid";
/// The NAT-traversal protocols a guardian must not speak.
const NAT_TRAVERSAL: [&str; 4] = [
    "/libp2p/autonat/1.0.0",
    "/libp2p/dcutr",
    "/libp2p/circuit/relay/0.2.0/stop",
    "/libp2p/circuit/relay/0.2.0/hop",
];

fn topics() -> NetworkTopics {
    NetworkTopics::for_network_cfg(&jetsam_chain::consensus::NetworkConfig::mainnet())
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn onion_settings(mode: OnionMode, tor: &FakeTor, socks: std::net::SocketAddr) -> OnionSettings {
    OnionConfig {
        enabled: true,
        mode,
        control: tor.addr.to_string(),
        socks: socks.to_string(),
        ..OnionConfig::default()
    }
    .validate()
    .unwrap()
    .unwrap()
}

struct Chain {
    chain: Arc<RwLock<MdbxChainContext>>,
    mempool: AsyncMempool,
}

fn chain(directory: &std::path::Path) -> Chain {
    let context = MdbxChainContext::open_or_create(directory).unwrap();
    let view = ChainView::from_mdbx(&context);
    Chain {
        chain: Arc::new(RwLock::new(context)),
        mempool: AsyncMempool::new(view, MempoolConfig::default()),
    }
}

#[allow(clippy::too_many_arguments)]
fn start(
    directory: &std::path::Path,
    chain: &Chain,
    listen: Vec<Multiaddr>,
    public: Vec<Multiaddr>,
    lan_discovery: bool,
    upnp: bool,
    onion: Option<OnionSettings>,
) -> anyhow::Result<P2PNetwork> {
    let (network, _task) = P2PNetwork::start_with_onion(
        listen,
        public,
        chain.chain.clone(),
        chain.mempool.clone(),
        topics(),
        PROOF_BANK,
        directory.to_path_buf(),
        BackgroundCapacity::Full,
        lan_discovery,
        upnp,
        None,
        onion,
    )?;
    Ok(network)
}

/// What a peer learned from one Identify message.
#[derive(Clone, Debug)]
struct Seen {
    from: PeerId,
    listen_addrs: Vec<Multiaddr>,
    protocols: Vec<String>,
}

/// A peer with the production behaviour, listening on loopback, answering
/// the network-profile handshake, recording every Identify it receives.
async fn observer() -> (PeerId, u16, Arc<Mutex<Vec<Seen>>>) {
    observer_with(None).await
}

/// [`observer`], announcing `external` as its own external address.
async fn observer_with(external: Option<Multiaddr>) -> (PeerId, u16, Arc<Mutex<Vec<Seen>>>) {
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
            NodeBehaviour::new(
                key,
                &protocol_id,
                relay_client,
                BackgroundCapacity::Full,
                false,
                false,
                false,
                false,
                false,
            )
        })
        .unwrap()
        .with_swarm_config(|cfg| cfg.with_idle_connection_timeout(Duration::from_secs(300)))
        .build();
    let port = free_port();
    swarm
        .listen_on(format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap())
        .unwrap();
    if let Some(external) = external {
        swarm.add_external_address(external);
    }
    let peer = *swarm.local_peer_id();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = seen.clone();
    tokio::spawn(async move {
        loop {
            match swarm.select_next_some().await {
                SwarmEvent::Behaviour(NodeBehaviourEvent::NetworkProfileSync(
                    request_response::Event::Message {
                        message: request_response::Message::Request { channel, .. },
                        ..
                    },
                )) => {
                    let _ = swarm.behaviour_mut().network_profile_sync.send_response(
                        channel,
                        NetworkProfileResponse {
                            profile: NetworkProfile::for_proof_bank(PROOF_BANK),
                        },
                    );
                }
                SwarmEvent::Behaviour(NodeBehaviourEvent::Identify(
                    identify::Event::Received { peer_id, info, .. },
                )) => {
                    let mut protocols = info
                        .protocols
                        .iter()
                        .map(|protocol| protocol.to_string())
                        .collect::<Vec<_>>();
                    protocols.sort();
                    record.lock().unwrap().push(Seen {
                        from: peer_id,
                        listen_addrs: info.listen_addrs,
                        protocols,
                    });
                }
                _ => {}
            }
        }
    });
    (peer, port, seen)
}

/// Wait until the observer holds an Identify satisfying `done`.
async fn identified(seen: &Arc<Mutex<Vec<Seen>>>, done: impl Fn(&Seen) -> bool) -> Seen {
    tokio::time::timeout(Duration::from_secs(90), async {
        loop {
            if let Some(found) = seen.lock().unwrap().iter().find(|s| done(s)).cloned() {
                return found;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await
    .expect("the expected Identify arrives")
}

async fn published(tor: &FakeTor) {
    tokio::time::timeout(Duration::from_secs(30), async {
        while !tor
            .received()
            .iter()
            .any(|line| line.starts_with("ADD_ONION"))
        {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await
    .expect("the node publishes its onion service");
    // The reply crosses the publication task, then the swarm task.
    tokio::time::sleep(Duration::from_millis(500)).await;
}

fn onion_addr() -> Multiaddr {
    format!("/onion3/{SYNTHETIC}:9700").parse().unwrap()
}

/// A plain TCP listener counting accepted connections.
async fn counting_listener() -> (u16, Arc<std::sync::atomic::AtomicUsize>) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let accepted = Arc::new(AtomicUsize::new(0));
    let count = accepted.clone();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            count.fetch_add(1, Ordering::SeqCst);
            drop(stream);
        }
    });
    (port, accepted)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn guardian_node_announces_only_its_onion_address_and_dials_only_through_tor() {
    let directory = tempfile::tempdir().unwrap();
    // This data directory ran on clearnet before: its identity is known.
    let clearnet_identity = crate::identity_store::load_or_create(directory.path())
        .unwrap()
        .public()
        .to_peer_id();
    let (observer_peer, observer_port, seen) = observer().await;
    let proxy = FakeSocks::start(FakeSocksConfig::forwarding(
        format!("127.0.0.1:{observer_port}").parse().unwrap(),
    ))
    .await;
    let tor = FakeTor::start(FakeTorConfig::null()).await;
    let chain = chain(directory.path());
    let guardian = start(
        directory.path(),
        &chain,
        Vec::new(),
        Vec::new(),
        false,
        false,
        Some(onion_settings(OnionMode::Guardian, &tor, proxy.addr)),
    )
    .expect("an onion-only configuration starts");
    published(&tor).await;

    guardian
        .dial(
            format!("/onion3/{REMOTE}:9700/p2p/{observer_peer}")
                .parse()
                .unwrap(),
        )
        .await;
    let first = identified(&seen, |s| s.listen_addrs.contains(&onion_addr())).await;
    assert_ne!(
        first.from, clearnet_identity,
        "the guardian reused the PeerId it had on clearnet"
    );
    // Pushes included: let them all arrive, then read every message.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let all = seen.lock().unwrap().clone();
    assert!(!all.is_empty());
    for message in &all {
        assert_eq!(message.from, first.from);
        // An Identify sent before the service was published carries no
        // address at all; none ever carries anything but the onion address.
        assert!(
            message
                .listen_addrs
                .iter()
                .all(|addr| *addr == onion_addr()),
            "the guardian announced {:?}",
            message.listen_addrs
        );
        for protocol in NAT_TRAVERSAL {
            assert!(
                !message.protocols.iter().any(|p| p == protocol),
                "the guardian speaks {protocol}"
            );
        }
    }
    let requests = proxy.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].domain().unwrap(), format!("{REMOTE}.onion"));
    assert_eq!(requests[0].port(), Some(9700));

    // Clearnet dials, by address and by name: no socket, no proxy use.
    let (port, accepted) = counting_listener().await;
    for addr in [
        format!("/ip4/127.0.0.1/tcp/{port}"),
        format!("/dns4/localhost/tcp/{port}"),
        format!("/dns/localhost/tcp/{port}/p2p/{}", PeerId::random()),
        format!("/ip4/127.0.0.1/tcp/{observer_port}/p2p/{observer_peer}"),
    ] {
        guardian.dial(addr.parse().unwrap()).await;
    }
    // Two automatic maintenance ticks.
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(
        accepted.load(std::sync::atomic::Ordering::SeqCst),
        0,
        "the guardian opened a clearnet socket"
    );
    assert_eq!(proxy.connections(), 1, "only the onion dial went to Tor");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn guardian_node_refuses_to_start_with_a_clearnet_exposure() {
    let directory = tempfile::tempdir().unwrap();
    let tor = FakeTor::start(FakeTorConfig::null()).await;
    let settings = onion_settings(OnionMode::Guardian, &tor, "127.0.0.1:9".parse().unwrap());
    let chain = chain(directory.path());
    let listen: Vec<Multiaddr> = vec!["/ip4/127.0.0.1/tcp/0".parse().unwrap()];
    let public: Vec<Multiaddr> = vec!["/ip4/8.8.8.8/tcp/9700".parse().unwrap()];
    for (listen, public, lan, upnp) in [
        (listen.clone(), Vec::new(), false, false),
        (Vec::new(), public.clone(), false, false),
        (Vec::new(), Vec::new(), true, false),
        (Vec::new(), Vec::new(), false, true),
    ] {
        let refused = start(
            directory.path(),
            &chain,
            listen.clone(),
            public.clone(),
            lan,
            upnp,
            Some(settings.clone()),
        );
        assert!(
            refused.is_err(),
            "started with listen={listen:?} public={public:?} lan={lan} upnp={upnp}"
        );
    }
    // Refused before anything was created or contacted.
    assert!(!directory
        .path()
        .join(crate::onion::guardian::GUARDIAN_KEY_DIR)
        .exists());
    assert_eq!(tor.connections(), 0);
}

/// Every protocol a node announced through Identify before D57, sorted
/// (mainnet topics, no client objects): read from an unmodified 23a3113 node
/// by the same observer (log `P7/onion/capture-pre-d57-protocols`).
const PRE_D57_PROTOCOLS: &[&str] = &[
    "/ipfs/id/1.0.0",
    "/ipfs/id/push/1.0.0",
    "/ipfs/ping/1.0.0",
    "/jetsam/mainnet/1.0.0/kad/1.0.0",
    "/jetsam/mainnet/1.0.0/sync/availability/1",
    "/jetsam/mainnet/1.0.0/sync/headers/4",
    "/jetsam/mainnet/1.0.0/sync/headers/5",
    "/jetsam/mainnet/1.0.0/sync/history-step/1",
    "/jetsam/mainnet/1.0.0/sync/manifest-page/1",
    "/jetsam/mainnet/1.0.0/sync/manifest/7",
    "/jetsam/mainnet/1.0.0/sync/mempool/3",
    "/jetsam/mainnet/1.0.0/sync/objects/2",
    "/jetsam/mainnet/1.0.0/sync/profile/6",
    "/jetsam/mainnet/1.0.0/sync/segment/5",
    "/libp2p/autonat/1.0.0",
    "/libp2p/circuit/relay/0.2.0/stop",
    "/meshsub/1.0.0",
    "/meshsub/1.1.0",
    "/meshsub/1.2.0",
];

/// Announced on top of [`PRE_D57_PROTOCOLS`] since libp2p 0.57 (D67, accepted
/// in D71): gossipsub 0.50 puts `/meshsub/1.3.0` first in its default
/// protocol list and has no public API to leave it out. The addition is
/// backwards compatible: multistream-select still settles on
/// `/meshsub/1.2.0` with a v1.6.0 node, which announces no 1.3.0 (measured in
/// D67 by an identify probe and the v1.6.0 interop replay I1). Nothing else
/// changed: every other identifier is the pre-D57 capture.
const ADDED_BY_GOSSIPSUB_0_50: &[&str] = &["/meshsub/1.3.0"];

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_node_without_onion_announces_exactly_what_it_announced_before() {
    let directory = tempfile::tempdir().unwrap();
    let chain = chain(directory.path());
    let port = free_port();
    let listen: Multiaddr = format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap();
    let (node, _task) = P2PNetwork::start(
        vec![listen.clone()],
        Vec::new(),
        chain.chain.clone(),
        chain.mempool.clone(),
        topics(),
        PROOF_BANK,
        directory.path().to_path_buf(),
        BackgroundCapacity::Full,
        false,
        false,
        None,
    )
    .unwrap();
    let (observer_peer, observer_port, seen_by_dialled) = observer().await;
    node.dial(
        format!("/ip4/127.0.0.1/tcp/{observer_port}/p2p/{observer_peer}")
            .parse()
            .unwrap(),
    )
    .await;
    let message = identified(&seen_by_dialled, |_| true).await;
    // The same identity as before, from the data directory itself.
    let stored = crate::identity_store::load_or_create(directory.path())
        .unwrap()
        .public()
        .to_peer_id();
    assert_eq!(message.from, stored);
    // Listen addresses are not hidden: the loopback listener is announced.
    assert!(
        message.listen_addrs.contains(&listen),
        "{:?}",
        message.listen_addrs
    );
    assert!(message.listen_addrs.iter().all(|addr| !matches!(
        addr.iter().next(),
        Some(libp2p::multiaddr::Protocol::Onion3(_))
    )));
    // Nothing onion-related is created on disk.
    assert!(!directory
        .path()
        .join(crate::onion::guardian::GUARDIAN_KEY_DIR)
        .exists());
    assert!(!directory
        .path()
        .join(crate::onion::key_store::ONION_KEY_FILE)
        .exists());
    println!("PROTOCOLS {:?}", message.protocols);
    // Read on a mainnet build; a testnet build names its own namespace.
    let mut expected = PRE_D57_PROTOCOLS
        .iter()
        .chain(ADDED_BY_GOSSIPSUB_0_50)
        .map(|protocol| protocol.replace("/jetsam/mainnet/1.0.0", &topics().protocol_id))
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(message.protocols, expected);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dual_node_announces_its_clearnet_and_onion_addresses_and_keeps_nat_traversal() {
    let directory = tempfile::tempdir().unwrap();
    let (observer_peer, observer_port, seen) = observer().await;
    let tor = FakeTor::start(FakeTorConfig::null()).await;
    let proxy = FakeSocks::start(FakeSocksConfig::forwarding("127.0.0.1:9".parse().unwrap())).await;
    let chain = chain(directory.path());
    let port = free_port();
    let listen: Multiaddr = format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap();
    let dual = start(
        directory.path(),
        &chain,
        vec![listen.clone()],
        Vec::new(),
        false,
        false,
        Some(onion_settings(OnionMode::Additional, &tor, proxy.addr)),
    )
    .unwrap();
    published(&tor).await;
    dual.dial(
        format!("/ip4/127.0.0.1/tcp/{observer_port}/p2p/{observer_peer}")
            .parse()
            .unwrap(),
    )
    .await;
    let message = identified(&seen, |s| s.listen_addrs.contains(&onion_addr())).await;
    assert!(
        message.listen_addrs.contains(&listen),
        "{:?}",
        message.listen_addrs
    );
    // The same identity as a clearnet node: dual mode does not hide it.
    let stored = crate::identity_store::load_or_create(directory.path())
        .unwrap()
        .public()
        .to_peer_id();
    assert_eq!(message.from, stored);
    // DCUtR is only offered on relayed connections, before D57 as now.
    for protocol in ["/libp2p/autonat/1.0.0", "/libp2p/circuit/relay/0.2.0/stop"] {
        assert!(
            message.protocols.iter().any(|p| p == protocol),
            "{protocol}"
        );
    }
    assert_eq!(proxy.connections(), 0, "a clearnet dial went through Tor");
}

// UPnP's behaviour needs a Tokio runtime to exist.
#[tokio::test]
async fn onion_only_behaviour_has_no_discovery_and_no_nat_traversal() {
    let key = libp2p::identity::Keypair::generate_ed25519();
    let protocol_id = topics().protocol_id;
    let guardian =
        NodeBehaviour::onion_only(&key, &protocol_id, BackgroundCapacity::Full, false, false)
            .unwrap();
    assert!(!guardian.relay_client.is_enabled());
    assert!(!guardian.relay_server.is_enabled());
    assert!(!guardian.dcutr.is_enabled());
    assert!(!guardian.autonat.is_enabled());
    assert!(!guardian.upnp.is_enabled());
    assert!(!guardian.mdns.is_enabled());

    // An ordinary node keeps every one it had.
    let (_, relay_client) = libp2p::relay::client::new(key.public().to_peer_id());
    let ordinary = NodeBehaviour::new(
        &key,
        &protocol_id,
        relay_client,
        BackgroundCapacity::Full,
        false,
        false,
        true,
        false,
        false,
    )
    .unwrap();
    assert!(ordinary.relay_client.is_enabled());
    assert!(ordinary.dcutr.is_enabled());
    assert!(ordinary.autonat.is_enabled());
    assert!(ordinary.upnp.is_enabled());
}

/// A third valid v3 address (public key 20 21 .. 3f, Python hashlib), which
/// the observer advertises as its own; never contacted.
const ADVERTISED: &str = "eaqseizeeutcokbjfivsyljof4ydcmrtgq2tmnzyhe5dwpb5hy74yhqd";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn guardian_node_keeps_onion_addresses_from_identify_and_serves_them_to_kademlia() {
    use crate::onion::swarm::{build_swarm, NodeTransports};
    use libp2p::kad;

    // A: an onion peer that advertises its onion address through Identify.
    let advertised: Multiaddr = format!("/onion3/{ADVERTISED}:9700").parse().unwrap();
    let (a_peer, a_port, a_seen) = observer_with(Some(advertised.clone())).await;
    let proxy = FakeSocks::start(FakeSocksConfig::forwarding(
        format!("127.0.0.1:{a_port}").parse().unwrap(),
    ))
    .await;
    let tor = FakeTor::start(FakeTorConfig::null()).await;
    let directory = tempfile::tempdir().unwrap();
    let chain = chain(directory.path());
    let guardian = start(
        directory.path(),
        &chain,
        Vec::new(),
        Vec::new(),
        false,
        false,
        Some(onion_settings(OnionMode::Guardian, &tor, proxy.addr)),
    )
    .unwrap();
    published(&tor).await;
    guardian
        .dial(
            format!("/onion3/{REMOTE}:9700/p2p/{a_peer}")
                .parse()
                .unwrap(),
        )
        .await;
    let guardian_peer = identified(&a_seen, |_| true).await.from;
    // Identify runs both ways: let the guardian take in A's addresses.
    tokio::time::sleep(Duration::from_secs(3)).await;

    // B reaches the guardian the way Tor would forward it: on the onion
    // service's loopback target, named in the ADD_ONION line.
    let target = tor
        .received()
        .into_iter()
        .find_map(|line| {
            line.strip_prefix("ADD_ONION ")?
                .split_once("Port=9700,")?
                .1
                .trim_end()
                .parse::<std::net::SocketAddr>()
                .ok()
        })
        .expect("the onion target");
    let b_proxy = FakeSocks::start(FakeSocksConfig::forwarding(
        format!("127.0.0.1:{a_port}").parse().unwrap(),
    ))
    .await;
    let protocol_id = topics().protocol_id;
    let mut b = build_swarm(
        libp2p::identity::Keypair::generate_ed25519(),
        NodeTransports::ClearnetAndOnion {
            socks: b_proxy.addr,
        },
        move |key, relay_client| {
            NodeBehaviour::new(
                key,
                &protocol_id,
                relay_client.expect("a relay client"),
                BackgroundCapacity::Full,
                false,
                false,
                false,
                false,
                false,
            )
        },
        |cfg| cfg.with_idle_connection_timeout(Duration::from_secs(60)),
    )
    .unwrap();
    b.behaviour_mut().kad.add_address(
        &guardian_peer,
        format!("/ip4/{}/tcp/{}", target.ip(), target.port())
            .parse()
            .unwrap(),
    );
    b.behaviour_mut().kad.get_closest_peers(PeerId::random());
    // The guardian answers with A and the addresses it keeps for A: B then
    // dials the advertised onion address through its own proxy.
    tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            if b_proxy
                .requests()
                .iter()
                .any(|request| request.domain() == Some(format!("{ADVERTISED}.onion")))
            {
                return;
            }
            tokio::select! {
                event = b.select_next_some() => {
                    if let SwarmEvent::Behaviour(NodeBehaviourEvent::Kad(
                        kad::Event::OutboundQueryProgressed { .. },
                    )) = event
                    {
                        b.behaviour_mut().kad.get_closest_peers(PeerId::random());
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {}
            }
        }
    })
    .await
    .expect("the guardian kept A's advertised onion address and served it");
}

/// Sockets of this process whose local or remote address is not loopback
/// (a listening socket's remote address is the unspecified address, which
/// is not an endpoint and is ignored). Linux only: `/proc/self/fd` and
/// `/proc/self/net/{tcp,tcp6,udp,udp6}`.
#[cfg(target_os = "linux")]
fn non_loopback_sockets_of_this_process() -> Vec<String> {
    use std::collections::HashSet;
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    let inodes: HashSet<String> = std::fs::read_dir("/proc/self/fd")
        .unwrap()
        .filter_map(|entry| std::fs::read_link(entry.ok()?.path()).ok())
        .filter_map(|target| {
            let target = target.to_string_lossy().into_owned();
            Some(
                target
                    .strip_prefix("socket:[")?
                    .strip_suffix(']')?
                    .to_string(),
            )
        })
        .collect();
    fn ip(hex: &str) -> Option<IpAddr> {
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).ok())
            .collect::<Option<Vec<u8>>>()?;
        // /proc prints each 32-bit word in host (little-endian) order.
        let words = bytes
            .chunks(4)
            .flat_map(|word| word.iter().rev().copied().collect::<Vec<_>>())
            .collect::<Vec<u8>>();
        match words.len() {
            4 => Some(IpAddr::V4(Ipv4Addr::new(
                words[0], words[1], words[2], words[3],
            ))),
            16 => Some(IpAddr::V6(Ipv6Addr::from(
                <[u8; 16]>::try_from(words).ok()?,
            ))),
            _ => None,
        }
    }
    let mut found = Vec::new();
    for table in ["tcp", "tcp6", "udp", "udp6"] {
        let Ok(text) = std::fs::read_to_string(format!("/proc/self/net/{table}")) else {
            continue;
        };
        for line in text.lines().skip(1) {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 10 || !inodes.contains(fields[9]) {
                continue;
            }
            let local = fields[1].split(':').next().and_then(ip);
            let remote = fields[2].split(':').next().and_then(ip);
            let local_ok = local.is_some_and(|ip| ip.is_loopback());
            let remote_ok = remote.is_some_and(|ip| ip.is_loopback() || ip.is_unspecified());
            if !local_ok || !remote_ok {
                found.push(format!("{table} {} -> {}", fields[1], fields[2]));
            }
        }
    }
    found
}

/// Real Tor, never contacted beyond this host: `DisableNetwork 1`, SOCKS and
/// ControlPort on 127.0.0.1, a temporary DataDirectory. Runs only with
/// `JETSAM_TOR_BIN=/path/to/tor` and `--ignored`. Checks the publication
/// through the real ControlPort (cookie authentication), the republication
/// with the stored key after Tor restarts, and that the guardian holds no
/// socket beyond loopback.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "needs a Tor binary in JETSAM_TOR_BIN; never reaches the Tor network"]
async fn real_tor_guardian_publishes_republishes_and_stays_on_loopback() {
    use std::os::unix::fs::PermissionsExt;
    let Some(tor_bin) = std::env::var_os("JETSAM_TOR_BIN") else {
        eprintln!("JETSAM_TOR_BIN unset: real Tor test not run");
        return;
    };
    let tor_dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(tor_dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let (socks_port, control_port) = (free_port(), free_port());
    let cookie = tor_dir.path().join("control.authcookie");
    let torrc = tor_dir.path().join("torrc");
    std::fs::write(
        &torrc,
        format!(
            "DataDirectory {data}\nDisableNetwork 1\nSocksPort 127.0.0.1:{socks_port}\n\
             ControlPort 127.0.0.1:{control_port}\nCookieAuthentication 1\n\
             CookieAuthFile {cookie}\nLog notice stderr\n",
            data = tor_dir.path().join("data").display(),
            cookie = cookie.display(),
        ),
    )
    .unwrap();
    let spawn_tor = || {
        std::process::Command::new(&tor_bin)
            .arg("-f")
            .arg(&torrc)
            .stdout(std::process::Stdio::null())
            .spawn()
            .expect("tor starts")
    };
    let control: std::net::SocketAddr = format!("127.0.0.1:{control_port}").parse().unwrap();
    let wait_control = || async move {
        tokio::time::timeout(Duration::from_secs(60), async {
            while tokio::net::TcpStream::connect(control).await.is_err() {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        })
        .await
        .expect("Tor's ControlPort comes up");
    };
    let mut tor = spawn_tor();
    wait_control().await;

    let settings = OnionConfig {
        enabled: true,
        mode: OnionMode::Guardian,
        control: control.to_string(),
        socks: format!("127.0.0.1:{socks_port}"),
        cookie_file: Some(cookie.clone()),
        ..OnionConfig::default()
    }
    .validate()
    .unwrap()
    .unwrap();
    let directory = tempfile::tempdir().unwrap();
    let chain = chain(directory.path());
    let guardian = start(
        directory.path(),
        &chain,
        Vec::new(),
        Vec::new(),
        false,
        false,
        Some(settings.clone()),
    )
    .unwrap();
    let key_dir = directory
        .path()
        .join(crate::onion::guardian::GUARDIAN_KEY_DIR);
    let stored = tokio::time::timeout(Duration::from_secs(60), async {
        loop {
            if let Ok(Some(key)) = crate::onion::key_store::load(&key_dir) {
                return key;
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    })
    .await
    .expect("Tor accepted ADD_ONION and the key was stored");

    // Clearnet dials while it runs: none may open a socket beyond loopback.
    let (port, accepted) = counting_listener().await;
    guardian
        .dial(format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap())
        .await;
    guardian
        .dial(format!("/dns4/localhost/tcp/{port}").parse().unwrap())
        .await;
    guardian
        .dial(
            format!("/onion3/{}:9700", stored.service_id())
                .parse()
                .unwrap(),
        )
        .await;
    tokio::time::sleep(Duration::from_secs(5)).await;
    assert_eq!(accepted.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(non_loopback_sockets_of_this_process(), Vec::<String>::new());

    // Tor restarts: the node publishes the same key again. A second
    // ADD_ONION of that key is then refused by Tor as a collision.
    tor.kill().unwrap();
    tor.wait().unwrap();
    let mut tor = spawn_tor();
    wait_control().await;
    let collision = tokio::time::timeout(Duration::from_secs(90), async {
        loop {
            tokio::time::sleep(Duration::from_secs(2)).await;
            let mut probe = crate::onion::control::TorControl::connect(control)
                .await
                .unwrap();
            let info = probe.protocol_info().await.unwrap();
            let choice = crate::onion::control::choose_auth(&info, &settings.auth).unwrap();
            probe.authenticate(&choice).await.unwrap();
            let attempt = probe
                .add_onion(
                    crate::onion::control::KeyRequest::Existing(&stored),
                    9700,
                    "127.0.0.1:9".parse().unwrap(),
                )
                .await;
            match attempt {
                Err(crate::onion::OnionError::ControlRefused { code: 550, .. }) => return,
                Ok(_) => {
                    // Not republished yet: withdraw ours and try again.
                    let _ = probe.del_onion(stored.service_id()).await;
                }
                Err(_) => {}
            }
        }
    })
    .await;
    assert!(
        collision.is_ok(),
        "the node never republished its stored key"
    );
    assert_eq!(
        crate::onion::key_store::load(&key_dir)
            .unwrap()
            .unwrap()
            .service_id(),
        stored.service_id(),
        "same key, same address"
    );
    assert_eq!(non_loopback_sockets_of_this_process(), Vec::<String>::new());
    tor.kill().unwrap();
    let _ = tor.wait();
}

/// Real Tor, never contacted beyond this host (`DisableNetwork 1`, ControlPort
/// on 127.0.0.1, no SOCKS port). The node authenticates by SAFECOOKIE: real
/// Tor accepts its client hash, so the cookie itself is never sent; with a
/// cookie Tor does not hold, the node sees from Tor's server hash that it
/// cannot prove anything, and answers nothing. Runs only with
/// `JETSAM_TOR_BIN=/path/to/tor` and `--ignored`.
#[cfg(target_os = "linux")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a Tor binary in JETSAM_TOR_BIN; never reaches the Tor network"]
async fn real_tor_authenticates_by_safecookie_and_detects_a_wrong_cookie() {
    use crate::onion::config::ControlAuthConfig;
    use crate::onion::control::{choose_auth, AuthChoice, AuthMethod, TorControl};
    use std::os::unix::fs::PermissionsExt;
    let Some(tor_bin) = std::env::var_os("JETSAM_TOR_BIN") else {
        eprintln!("JETSAM_TOR_BIN unset: real Tor test not run");
        return;
    };
    let tor_dir = tempfile::tempdir().unwrap();
    std::fs::set_permissions(tor_dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
    let control_port = free_port();
    let cookie = tor_dir.path().join("control.authcookie");
    let torrc = tor_dir.path().join("torrc");
    std::fs::write(
        &torrc,
        format!(
            "DataDirectory {data}\nDisableNetwork 1\nSocksPort 0\n\
             ControlPort 127.0.0.1:{control_port}\nCookieAuthentication 1\n\
             CookieAuthFile {cookie}\nLog notice stderr\n",
            data = tor_dir.path().join("data").display(),
            cookie = cookie.display(),
        ),
    )
    .unwrap();
    let mut tor = std::process::Command::new(&tor_bin)
        .arg("-f")
        .arg(&torrc)
        .stdout(std::process::Stdio::null())
        .spawn()
        .expect("tor starts");
    let control: std::net::SocketAddr = format!("127.0.0.1:{control_port}").parse().unwrap();
    tokio::time::timeout(Duration::from_secs(60), async {
        while tokio::net::TcpStream::connect(control).await.is_err() || !cookie.exists() {
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    })
    .await
    .expect("Tor's ControlPort comes up");

    let auth = ControlAuthConfig {
        cookie_file: Some(cookie.clone()),
        password_file: None,
    };
    let mut session = TorControl::connect(control).await.unwrap();
    let info = session.protocol_info().await.unwrap();
    assert!(
        info.methods.contains(&AuthMethod::SafeCookie),
        "{:?}",
        info.methods
    );
    assert_eq!(info.cookie_file.as_deref(), Some(cookie.as_path()));
    let choice = choose_auth(&info, &auth).unwrap();
    assert_eq!(choice, AuthChoice::SafeCookie(cookie.clone()));
    session.authenticate(&choice).await.unwrap();
    // Authenticated: Tor answers what it refuses (514) before.
    let version = session.command("GETINFO version").await.unwrap();
    assert!(version.lines[0].text.starts_with("version="), "{version:?}");

    // A cookie Tor does not hold: Tor's server hash cannot match it.
    let wrong = tor_dir.path().join("not-tor-s.cookie");
    let mut bytes = std::fs::read(&cookie).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&wrong, &bytes).unwrap();
    let mut session = TorControl::connect(control).await.unwrap();
    session.protocol_info().await.unwrap();
    assert!(matches!(
        session.authenticate(&AuthChoice::SafeCookie(wrong)).await,
        Err(crate::onion::OnionError::Auth(_))
    ));
    // Nothing was answered, so that session never became authenticated.
    assert!(session.command("GETINFO version").await.is_err());
    tor.kill().unwrap();
    let _ = tor.wait();
}
