// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Two-node round trip of the v1.5 client-object protocol (M3 task 3.7):
//! real libp2p swarms (TCP on the loopback, Noise, Yamux), the production
//! behaviour and codec. Eight client proofs of measured size requested at
//! once; then a peer serving altered bytes next to an honest one, through
//! the requester's fetch policy: the altered bundle is refused, the liar is
//! the only peer reported, the honest copy is fetched.

use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use futures::{AsyncRead, AsyncWrite, AsyncWriteExt, StreamExt};
use jetsam_chain::consensus::client_objects::{ClientObject, ClientSubmission};
use jetsam_p2p::client_object_codec::{
    client_object_behaviour, ClientObjectCodec, ClientObjectRequest, ClientObjectResponse,
    CLIENT_OBJECTS_PROTOCOL_SUFFIX,
};
use jetsam_p2p::client_object_protocol::{
    ClientProofAnnouncement, ClientProofBundle, ClientProofFetcher,
};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, PagedSpendIntent, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};
use libp2p::{
    noise,
    request_response::{self, Codec, Event, Message, ProtocolSupport},
    swarm::{StreamProtocol, SwarmEvent},
    tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder,
};

const NETWORK: &str = "/jetsam-client-transport-test";
/// The example client's proof on the wire (M2 task 2.6, measured).
const MEASURED_PROOF_BYTES: usize = 408_721;

fn bundle(io: u8) -> ClientProofBundle {
    let submission = ClientSubmission {
        matrix_digest: [0x87; 32],
        io_commitment: [io; 32],
    };
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: 1_000 + u32::from(io) * 4,
        amount: 2_000_000,
        creation_id: 1,
    };
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput {
        slot_index: 1_001 + u32::from(io) * 4,
        amount: 0,
        owner: ClientObject::Submission(submission).marker(),
    };
    let payment = PagedSpendIntent::new(
        vec![TxPage {
            body: TxBody {
                epoch_anchor: [7u8; 32],
                fee: 2_000_000,
                input_owner: Address([0x51; 32]),
                inputs,
                outputs,
                validity_bitmap: 1
                    | output_bitmap_bit(0)
                    | PAGED_SPEND_START_BIT
                    | PAGED_SPEND_END_BIT,
                is_coinbase: false,
            },
        }],
        vec![0xA5; 4_096],
    )
    .unwrap();
    let proof = (0..MEASURED_PROOF_BYTES)
        .map(|index| (index as u8) ^ io)
        .collect();
    ClientProofBundle::new(submission, payment, proof).unwrap()
}

fn swarm_with<B: libp2p::swarm::NetworkBehaviour>(behaviour: B) -> Swarm<B> {
    SwarmBuilder::with_new_identity()
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )
        .unwrap()
        .with_behaviour(|_| behaviour)
        .unwrap()
        .with_swarm_config(|config| config.with_idle_connection_timeout(Duration::from_secs(60)))
        .build()
}

async fn listen<B: libp2p::swarm::NetworkBehaviour>(swarm: &mut Swarm<B>) -> Multiaddr {
    swarm
        .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
        .unwrap();
    loop {
        if let SwarmEvent::NewListenAddr { address, .. } = swarm.select_next_some().await {
            return address;
        }
    }
}

/// An honest node: serves the bundles it holds, `None` otherwise.
async fn spawn_server(objects: BTreeMap<ClientObjectRequest, Vec<u8>>) -> (PeerId, Multiaddr) {
    let mut swarm = swarm_with(client_object_behaviour(NETWORK).unwrap());
    let address = listen(&mut swarm).await;
    let peer = *swarm.local_peer_id();
    tokio::spawn(async move {
        loop {
            if let SwarmEvent::Behaviour(Event::Message {
                message:
                    Message::Request {
                        request, channel, ..
                    },
                ..
            }) = swarm.select_next_some().await
            {
                let response = match objects.get(&request) {
                    Some(bytes) => ClientObjectResponse::ready(request, bytes.clone()),
                    None => ClientObjectResponse::unavailable(request),
                };
                let _ = swarm.behaviour_mut().send_response(channel, response);
            }
        }
    });
    (peer, address)
}

/// A lying node's codec: it answers with any frame it likes. Here, the
/// honest frame of the requested bundle with one payload byte flipped.
#[derive(Clone, Default)]
struct LyingCodec;

#[async_trait]
impl Codec for LyingCodec {
    type Protocol = StreamProtocol;
    type Request = ClientObjectRequest;
    type Response = Vec<u8>;

    async fn read_request<T>(
        &mut self,
        protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        ClientObjectCodec::default()
            .read_request(protocol, io)
            .await
    }

    async fn read_response<T>(
        &mut self,
        _: &Self::Protocol,
        _: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        Err(io::Error::other("the liar never requests"))
    }

    async fn write_request<T>(
        &mut self,
        _: &Self::Protocol,
        _: &mut T,
        _: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        Err(io::Error::other("the liar never requests"))
    }

    async fn write_response<T>(
        &mut self,
        _: &Self::Protocol,
        io: &mut T,
        frame: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        io.write_all(&frame).await?;
        io.flush().await
    }
}

async fn spawn_liar(objects: BTreeMap<ClientObjectRequest, Vec<u8>>) -> (PeerId, Multiaddr) {
    let protocol =
        StreamProtocol::try_from_owned(format!("{NETWORK}{CLIENT_OBJECTS_PROTOCOL_SUFFIX}"))
            .unwrap();
    let behaviour = request_response::Behaviour::with_codec(
        LyingCodec,
        [(protocol.clone(), ProtocolSupport::Full)],
        request_response::Config::default(),
    );
    let mut swarm = swarm_with(behaviour);
    let address = listen(&mut swarm).await;
    let peer = *swarm.local_peer_id();
    tokio::spawn(async move {
        loop {
            if let SwarmEvent::Behaviour(Event::Message {
                message:
                    Message::Request {
                        request, channel, ..
                    },
                ..
            }) = swarm.select_next_some().await
            {
                let mut frame = futures::io::Cursor::new(Vec::new());
                ClientObjectCodec::default()
                    .write_response(
                        &protocol,
                        &mut frame,
                        ClientObjectResponse::ready(request, objects[&request].clone()),
                    )
                    .await
                    .unwrap();
                let mut frame = frame.into_inner();
                let last = frame.len() - 1;
                frame[last] ^= 1;
                let _ = swarm.behaviour_mut().send_response(channel, frame);
            }
        }
    });
    (peer, address)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn eight_client_proofs_cross_two_nodes_at_once() {
    let bundles: Vec<ClientProofBundle> = (0..8).map(bundle).collect();
    let objects: BTreeMap<_, _> = bundles
        .iter()
        .map(|bundle| (ClientObjectRequest::Proof(bundle.id()), bundle.encode()))
        .collect();
    let total_bytes: usize = objects.values().map(Vec::len).sum();
    let (server, address) = spawn_server(objects).await;

    // Ids first: hashing is CPU work that must not starve the swarm loop.
    let ids: Vec<_> = bundles.iter().map(ClientProofBundle::id).collect();
    let mut client = swarm_with(client_object_behaviour(NETWORK).unwrap());
    // As on the network: the proofs were announced by a connected peer.
    // (Requests sent before any connection race one dial each.)
    client.dial(address).unwrap();
    loop {
        match client.select_next_some().await {
            SwarmEvent::ConnectionEstablished { peer_id, .. } if peer_id == server => break,
            SwarmEvent::OutgoingConnectionError { error, .. } => panic!("dial failed: {error}"),
            _ => {}
        }
    }
    let started = Instant::now();
    let mut pending = BTreeMap::new();
    for id in ids {
        let request_id = client
            .behaviour_mut()
            .send_request(&server, ClientObjectRequest::Proof(id));
        pending.insert(request_id, id);
    }
    let mut received = Vec::new();
    let deadline = tokio::time::sleep(Duration::from_secs(120));
    tokio::pin!(deadline);
    while !pending.is_empty() {
        tokio::select! {
            _ = &mut deadline => panic!("{} requests still pending", pending.len()),
            event = client.select_next_some() => match event {
                SwarmEvent::Behaviour(Event::Message {
                    message: Message::Response { request_id, response },
                    ..
                }) => {
                    let id = pending.remove(&request_id).expect("one response per request");
                    assert_eq!(response.request, ClientObjectRequest::Proof(id));
                    let bytes = response.bytes.expect("the server holds every bundle");
                    received.push(ClientProofBundle::decode_for(&id, &bytes).unwrap());
                }
                SwarmEvent::Behaviour(Event::OutboundFailure { error, .. }) => {
                    panic!("request failed: {error}")
                }
                _ => {}
            }
        }
    }
    let elapsed = started.elapsed();
    eprintln!(
        "8 client proof bundles, {total_bytes} bytes, two nodes on the loopback: {:.3} s",
        elapsed.as_secs_f64()
    );
    received.sort_by_key(|bundle| bundle.submission.io_commitment);
    assert_eq!(received, bundles);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn an_altered_proof_is_refused_and_only_its_server_is_reported() {
    let bundle = bundle(1);
    let id = bundle.id();
    let objects: BTreeMap<_, _> = [(ClientObjectRequest::Proof(id), bundle.encode())].into();
    let (liar, liar_address) = spawn_liar(objects.clone()).await;
    let (honest, honest_address) = spawn_server(objects).await;

    let mut client = swarm_with(client_object_behaviour(NETWORK).unwrap());
    client.add_peer_address(liar, liar_address);
    client.add_peer_address(honest, honest_address);
    let mut fetcher = ClientProofFetcher::new(16, 4, 2);
    let announcement = ClientProofAnnouncement { id, fee: 2_000_000 };
    // The liar announced first: it is asked first.
    assert!(fetcher.announced(liar, announcement));
    assert!(fetcher.announced(honest, announcement));

    let mut reported = BTreeSet::new();
    let mut fetched = None;
    let mut in_flight = BTreeMap::new();
    let deadline = tokio::time::sleep(Duration::from_secs(120));
    tokio::pin!(deadline);
    while fetched.is_none() {
        for (peer, id) in fetcher.next_requests() {
            let request_id = client
                .behaviour_mut()
                .send_request(&peer, ClientObjectRequest::Proof(id));
            in_flight.insert(request_id, (peer, id));
        }
        tokio::select! {
            _ = &mut deadline => panic!("the proof was never fetched"),
            event = client.select_next_some() => match event {
                SwarmEvent::Behaviour(Event::Message {
                    message: Message::Response { request_id, response },
                    ..
                }) => {
                    let (peer, id) = in_flight.remove(&request_id).unwrap();
                    match response.bytes.map(|bytes| ClientProofBundle::decode_for(&id, &bytes)) {
                        Some(Ok(bundle)) => {
                            fetcher.completed(&peer, &id);
                            fetched = Some((peer, bundle));
                        }
                        Some(Err(_)) => {
                            if fetcher.failed(&peer, &id, true) {
                                reported.insert(peer);
                            }
                        }
                        None => {
                            fetcher.failed(&peer, &id, false);
                        }
                    }
                }
                SwarmEvent::Behaviour(Event::OutboundFailure { request_id, error, .. }) => {
                    let (peer, id) = in_flight.remove(&request_id).unwrap();
                    // The codec refused the payload: the bytes were not the
                    // object. Anything else (timeout, closed) is no proof of
                    // malice.
                    let wrong_bytes = matches!(error, request_response::OutboundFailure::Io(_));
                    if fetcher.failed(&peer, &id, wrong_bytes) {
                        reported.insert(peer);
                    }
                }
                _ => {}
            }
        }
    }
    let (from, received) = fetched.unwrap();
    assert_eq!(from, honest);
    assert_eq!(received, bundle);
    assert_eq!(reported, BTreeSet::from([liar]));
    assert!(!fetcher.is_wanted(&id));
}
