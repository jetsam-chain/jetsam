// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! M3.8: the client-object protocol and announcement topic through the real
//! P2P reactor (`P2PNetwork::start`, the node's `NodeBehaviour`): two nodes on
//! loopback, one holding a client proof bundle and a matrix chunk.

use std::sync::Arc;
use std::time::Duration;

use jetsam_chain::consensus::client_objects::ClientSubmission;
use jetsam_chain::storage::MdbxChainContext;
use jetsam_mempool::{AsyncMempool, ChainView, MempoolConfig};
use jetsam_p2p::client_object_codec::ClientObjectRequest;
use jetsam_p2p::client_object_protocol::{ClientProofAnnouncement, ClientProofId};
use jetsam_p2p::client_object_transport::ClientObjectSource;
use jetsam_p2p::network::{NetworkEvent, RequestFailureKind};
use jetsam_p2p::{NetworkCommand, NetworkTopics, P2PNetwork};
use libp2p::{Multiaddr, PeerId};
use tokio::sync::RwLock;

struct Holder {
    id: ClientProofId,
    bundle: Vec<u8>,
}

/// A node with client objects (a v1.5 relation armed) that holds none yet.
struct HoldsNothing;

impl ClientObjectSource for HoldsNothing {
    fn client_object(&self, _request: &ClientObjectRequest) -> Option<Vec<u8>> {
        None
    }
}

impl ClientObjectSource for Holder {
    fn client_object(&self, request: &ClientObjectRequest) -> Option<Vec<u8>> {
        match request {
            ClientObjectRequest::Proof(id) if *id == self.id => Some(self.bundle.clone()),
            _ => None,
        }
    }
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

async fn start(
    directory: &std::path::Path,
    port: u16,
    source: Option<Arc<dyn ClientObjectSource>>,
) -> P2PNetwork {
    let context = MdbxChainContext::open_or_create(directory).unwrap();
    let view = ChainView::from_mdbx(&context);
    let chain = Arc::new(RwLock::new(context));
    let mempool = AsyncMempool::new(view, MempoolConfig::default());
    let topics =
        NetworkTopics::for_network_cfg(&jetsam_chain::consensus::NetworkConfig::mainnet());
    let listen: Multiaddr = format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap();
    let (network, _task) = P2PNetwork::start(
        vec![listen],
        Vec::new(),
        chain,
        mempool,
        topics,
        [0x7A; 32],
        directory.to_path_buf(),
        jetsam_p2p::BackgroundCapacity::Full,
        false,
        false,
        source,
    )
    .unwrap();
    network
}

async fn next_matching<T>(
    events: &mut jetsam_p2p::network::NetworkEventReceiver,
    timeout: Duration,
    mut pick: impl FnMut(NetworkEvent) -> Option<T>,
) -> Option<T> {
    tokio::time::timeout(timeout, async {
        loop {
            if let Ok(event) = events.recv().await {
                if let Some(found) = pick(event) {
                    return found;
                }
            }
        }
    })
    .await
    .ok()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_reactors_serve_fetch_and_announce_client_objects() {
    let submission = ClientSubmission {
        matrix_digest: [0xD1; 32],
        io_commitment: [0x10; 32],
    };
    // An opaque bundle: the reactor only checks the bytes are the id's.
    let bundle: Vec<u8> = (0..4_096u32).map(|index| (index * 7) as u8).collect();
    let mut bundle = bundle;
    bundle[..4].copy_from_slice(b"JCP1");
    let id = ClientProofId::of_bytes(submission, &bundle).unwrap();
    let holder_dir = tempfile::tempdir().unwrap();
    let asker_dir = tempfile::tempdir().unwrap();
    let holder_port = free_port();
    let holder = start(
        holder_dir.path(),
        holder_port,
        Some(Arc::new(Holder {
            id,
            bundle: bundle.clone(),
        })),
    )
    .await;
    let asker = start(asker_dir.path(), free_port(), Some(Arc::new(HoldsNothing))).await;
    let mut asker_events = asker.subscribe();

    asker
        .dial(format!("/ip4/127.0.0.1/tcp/{holder_port}").parse().unwrap())
        .await;
    let holder_peer: PeerId = next_matching(&mut asker_events, Duration::from_secs(30), |event| {
        match event {
            NetworkEvent::PeerConnected { peer, .. } => Some(peer),
            _ => None,
        }
    })
    .await
    .expect("the two nodes connect");

    // The holder serves the exact bundle.
    asker
        .cmd_tx
        .send(NetworkCommand::FetchClientObject {
            token: 7,
            peer: holder_peer,
            request: ClientObjectRequest::Proof(id),
        })
        .await
        .unwrap();
    let fetched = next_matching(&mut asker_events, Duration::from_secs(60), |event| match event {
        NetworkEvent::ClientObjectFetched { token: 7, bytes, .. } => Some(bytes.to_vec()),
        NetworkEvent::ClientObjectFetchFailed { token: 7, kind, .. } => {
            panic!("fetch failed: {kind:?}")
        }
        _ => None,
    })
    .await
    .expect("the bundle arrives");
    assert_eq!(fetched, bundle);

    // An object it does not hold: unavailable, not a lie.
    let other = ClientProofId::of_bytes(submission, &bundle[..2_048]).unwrap();
    asker
        .cmd_tx
        .send(NetworkCommand::FetchClientObject {
            token: 8,
            peer: holder_peer,
            request: ClientObjectRequest::Proof(other),
        })
        .await
        .unwrap();
    let kind = next_matching(&mut asker_events, Duration::from_secs(60), |event| match event {
        NetworkEvent::ClientObjectFetchFailed { token: 8, kind, .. } => Some(kind),
        NetworkEvent::ClientObjectFetched { token: 8, .. } => panic!("served an unheld object"),
        _ => None,
    })
    .await
    .expect("an answer");
    assert_eq!(kind, RequestFailureKind::Unavailable);

    // A peer without a connection is never dialled for a request.
    asker
        .cmd_tx
        .send(NetworkCommand::FetchClientObject {
            token: 9,
            peer: PeerId::random(),
            request: ClientObjectRequest::Proof(id),
        })
        .await
        .unwrap();
    let kind = next_matching(&mut asker_events, Duration::from_secs(10), |event| match event {
        NetworkEvent::ClientObjectFetchFailed { token: 9, kind, .. } => Some(kind),
        _ => None,
    })
    .await
    .expect("an immediate failure");
    assert_eq!(kind, RequestFailureKind::ConnectionClosed);

    // The holder's announcement reaches the asker over gossip once the
    // mesh has formed; it is repeated until it does.
    let announcement = ClientProofAnnouncement { id, fee: 2_000_000 };
    let mut received = None;
    for attempt in 0..30u64 {
        let mut fresh = announcement;
        fresh.fee += attempt; // distinct bytes: gossip deduplicates by content
        holder
            .cmd_tx
            .send(NetworkCommand::AnnounceClientProof {
                announcement: fresh,
            })
            .await
            .unwrap();
        received = next_matching(&mut asker_events, Duration::from_secs(2), |event| match event {
            NetworkEvent::ClientProofAnnounced {
                from, announcement, ..
            } => Some((from, announcement)),
            _ => None,
        })
        .await;
        if received.is_some() {
            break;
        }
    }
    let (from, heard) = received.expect("the announcement is gossiped");
    assert_eq!(from, holder_peer);
    assert_eq!(heard.id, id);
}

/// The protocols `node` (listening on `port`) advertises to a plain
/// identify-only probe that dials it.
async fn advertised_protocols(port: u16) -> Vec<String> {
    use futures::StreamExt;
    use libp2p::{identify, noise, swarm::SwarmEvent, tcp, yamux, SwarmBuilder};
    let mut probe = SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(tcp::Config::default(), noise::Config::new, yamux::Config::default)
        .unwrap()
        .with_behaviour(|key| {
            identify::Behaviour::new(identify::Config::new("/jetsam/1.0.0".into(), key.public()))
        })
        .unwrap()
        .build();
    let address: Multiaddr = format!("/ip4/127.0.0.1/tcp/{port}").parse().unwrap();
    probe.dial(address.clone()).unwrap();
    tokio::time::timeout(Duration::from_secs(120), async {
        loop {
            match probe.select_next_some().await {
                // The node's listener may not be up yet, or the node may close
                // the connection (the probe lacks the sync protocols) before
                // its own identify reached the probe under load: dial again.
                SwarmEvent::OutgoingConnectionError { .. } | SwarmEvent::ConnectionClosed { .. } => {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    probe.dial(address.clone()).unwrap();
                }
                SwarmEvent::Behaviour(identify::Event::Received { info, .. }) => {
                    return info
                    .protocols
                    .iter()
                    .map(|protocol| protocol.to_string())
                    .collect::<Vec<_>>();
                }
                _ => {}
            }
        }
    })
    .await
    .expect("the node identifies itself to the probe")
}

/// M3.10 (§8 trou d): a node without v1.5 client objects (the dormant
/// release: no v1.5 pack, no clock) does not expose `/client/objects/1` — what
/// it advertises is what it advertised before v1.5 — while a node with them
/// does (the positive control that the probe would see it).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dormant_node_does_not_advertise_the_client_object_protocol() {
    let dormant_dir = tempfile::tempdir().unwrap();
    let armed_dir = tempfile::tempdir().unwrap();
    let (dormant_port, armed_port) = (free_port(), free_port());
    let _dormant = start(dormant_dir.path(), dormant_port, None).await;
    let _armed = start(armed_dir.path(), armed_port, Some(Arc::new(HoldsNothing))).await;
    let client_protocol = |protocols: &[String]| {
        protocols
            .iter()
            .any(|protocol| protocol.ends_with("/client/objects/1"))
    };
    let armed = advertised_protocols(armed_port).await;
    assert!(client_protocol(&armed), "positive control: {armed:?}");
    let dormant = advertised_protocols(dormant_port).await;
    assert!(
        dormant.iter().any(|protocol| protocol.ends_with("/sync/objects/2")),
        "the probe reads the dormant node's protocols: {dormant:?}"
    );
    assert!(!client_protocol(&dormant), "dormant node advertises: {dormant:?}");
}

/// M3.10 (§8 gap b): a registration one node holds reaches another node over
/// the client topic, whole (registration and paying transaction), so that any
/// miner can include it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_held_registration_is_relayed_to_every_node() {
    use jetsam_chain::consensus::client_objects::{
        ClientObject, ClientObjectRules, ClientRegistration, CLIENT_LICENSE_BURN_ADDRESS,
    };
    use jetsam_p2p::client_object_protocol::ClientRegistrationNotice;
    use jetsam_poseidon2b::primitives::Address;
    use jetsam_tx::{
        output_bitmap_bit, PagedSpendIntent, TxBody, TxInput, TxOutput, TxPage,
        PAGED_SPEND_END_BIT, PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
    };
    let rules = ClientObjectRules::current();
    let registration = ClientRegistration {
        matrix_digest: [0x31; 32],
        matrix_file_root: [0x32; 32],
        matrix_file_len: 4_096,
    };
    let marker = ClientObject::Registration(registration).marker();
    let license = rules.destination.split(rules.license_micro).burn;
    let notice_paying_fee = |fee: u64| {
        let mut inputs = [TxInput::dummy(); TX_INPUTS];
        inputs[0] = TxInput {
            slot_index: 300,
            amount: license + fee,
            creation_id: 1,
        };
        let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
        outputs[0] = TxOutput {
            slot_index: 301,
            amount: license,
            owner: CLIENT_LICENSE_BURN_ADDRESS,
        };
        outputs[1] = TxOutput {
            slot_index: 302,
            amount: 0,
            owner: marker,
        };
        let payment = PagedSpendIntent::new(
            vec![TxPage {
                body: TxBody {
                    epoch_anchor: [7u8; 32],
                    fee,
                    input_owner: Address([0x51; 32]),
                    inputs,
                    outputs,
                    validity_bitmap: 1
                        | output_bitmap_bit(0)
                        | output_bitmap_bit(1)
                        | PAGED_SPEND_START_BIT
                        | PAGED_SPEND_END_BIT,
                    is_coinbase: false,
                },
            }],
            vec![0xA5; 64],
        )
        .unwrap();
        ClientRegistrationNotice::new(registration, payment, &rules).unwrap()
    };
    let holder_dir = tempfile::tempdir().unwrap();
    let asker_dir = tempfile::tempdir().unwrap();
    let holder_port = free_port();
    let holder = start(holder_dir.path(), holder_port, Some(Arc::new(HoldsNothing))).await;
    let asker = start(asker_dir.path(), free_port(), Some(Arc::new(HoldsNothing))).await;
    let mut asker_events = asker.subscribe();
    asker
        .dial(format!("/ip4/127.0.0.1/tcp/{holder_port}").parse().unwrap())
        .await;
    let holder_peer: PeerId = next_matching(&mut asker_events, Duration::from_secs(30), |event| {
        match event {
            NetworkEvent::PeerConnected { peer, .. } => Some(peer),
            _ => None,
        }
    })
    .await
    .expect("the two nodes connect");

    // Repeated (with distinct bytes: gossip deduplicates by content) until
    // the mesh has formed.
    let mut received = None;
    for attempt in 0..30u64 {
        let notice = Arc::new(notice_paying_fee(2_000_000 + attempt));
        holder
            .cmd_tx
            .send(NetworkCommand::AnnounceClientRegistration {
                notice: Arc::clone(&notice),
            })
            .await
            .unwrap();
        received = next_matching(&mut asker_events, Duration::from_secs(2), |event| match event {
            NetworkEvent::ClientRegistrationRelayed { relayed_by, notice } => {
                Some((relayed_by, notice))
            }
            _ => None,
        })
        .await;
        if received.is_some() {
            break;
        }
    }
    let (relayed_by, heard) = received.expect("the registration is relayed");
    assert_eq!(relayed_by, holder_peer);
    assert_eq!(heard.registration, registration);
    assert_eq!(heard.payment.pages[0].body.outputs[1].owner, marker);
}
