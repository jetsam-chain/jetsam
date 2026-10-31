// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The transports a node's swarm is built with, by onion mode.
//!
//! - Onion disabled: TCP, DNS, relay client — the pre-D57 chain, call for
//!   call.
//! - Dual: the same, with the onion dial transport between TCP and DNS (TCP
//!   refuses `/onion3`, the onion transport refuses everything else, and the
//!   DNS layer has nothing to resolve in an onion address).
//! - Guardian: no TCP dialler, no DNS, no relay client. Outbound, only the
//!   onion dial transport exists, so every other address fails with
//!   `MultiaddrNotSupported` before any socket is opened or any name looked
//!   up. Inbound, libp2p can only listen through a transport: a
//!   [`LoopbackListener`] binds the loopback target Tor forwards to, accepts
//!   no other listen address, and refuses every dial.

use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

use libp2p::core::muxing::StreamMuxerBox;
use libp2p::core::transport::{Boxed, DialOpts, ListenerId, TransportError, TransportEvent};
use libp2p::core::{upgrade, Transport};
use libp2p::identity::Keypair;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{
    multiaddr::Protocol, noise, relay, tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder,
};

use super::config::{OnionMode, OnionSettings};
use super::transport::OnionDialTransport;

/// The transport set of a node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeTransports {
    Clearnet,
    ClearnetAndOnion { socks: SocketAddr },
    OnionOnly { socks: SocketAddr },
}

impl NodeTransports {
    pub fn of(onion: Option<&OnionSettings>) -> Self {
        match onion {
            None => Self::Clearnet,
            Some(settings) if settings.mode == OnionMode::Guardian => Self::OnionOnly {
                socks: settings.socks,
            },
            Some(settings) => Self::ClearnetAndOnion {
                socks: settings.socks,
            },
        }
    }
}

/// A transport that only listens, and only on a loopback IP/TCP address:
/// the target of the onion service. It never dials.
#[derive(Debug)]
pub struct LoopbackListener<T>(pub T);

fn is_loopback_tcp(addr: &Multiaddr) -> bool {
    let mut protocols = addr.iter();
    let loopback = match protocols.next() {
        Some(Protocol::Ip4(ip)) => ip.is_loopback(),
        Some(Protocol::Ip6(ip)) => ip.is_loopback(),
        _ => false,
    };
    loopback && matches!(protocols.next(), Some(Protocol::Tcp(_))) && protocols.next().is_none()
}

impl<T: Transport + Unpin> Transport for LoopbackListener<T> {
    type Output = T::Output;
    type Error = T::Error;
    type ListenerUpgrade = T::ListenerUpgrade;
    type Dial = T::Dial;

    fn listen_on(
        &mut self,
        id: ListenerId,
        addr: Multiaddr,
    ) -> Result<(), TransportError<Self::Error>> {
        if !is_loopback_tcp(&addr) {
            return Err(TransportError::MultiaddrNotSupported(addr));
        }
        self.0.listen_on(id, addr)
    }

    fn remove_listener(&mut self, id: ListenerId) -> bool {
        self.0.remove_listener(id)
    }

    fn dial(
        &mut self,
        addr: Multiaddr,
        opts: DialOpts,
    ) -> Result<Self::Dial, TransportError<Self::Error>> {
        // Never: a guardian's only way out is Tor.
        let _ = opts;
        Err(TransportError::MultiaddrNotSupported(addr))
    }

    fn poll(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        Pin::new(&mut self.get_mut().0).poll(cx)
    }
}

/// Build the node's swarm. `behaviour` receives the relay client behaviour
/// when the transport set has one (never in guardian mode).
pub fn build_swarm<B, F>(
    identity: Keypair,
    transports: NodeTransports,
    behaviour: F,
    config: impl FnOnce(libp2p::swarm::Config) -> libp2p::swarm::Config,
) -> anyhow::Result<Swarm<B>>
where
    B: NetworkBehaviour,
    F: FnOnce(
        &Keypair,
        Option<relay::client::Behaviour>,
    ) -> Result<B, Box<dyn std::error::Error + Send + Sync>>,
{
    let builder = SwarmBuilder::with_existing_identity(identity).with_tokio();
    Ok(match transports {
        NodeTransports::Clearnet => builder
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_dns()?
            .with_relay_client(noise::Config::new, yamux::Config::default)?
            .with_behaviour(|key, relay_client| behaviour(key, Some(relay_client)))?
            .with_swarm_config(config)
            .build(),
        NodeTransports::ClearnetAndOnion { socks } => builder
            .with_tcp(
                tcp::Config::default().nodelay(true),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_other_transport(|key| OnionDialTransport::new(socks).authenticated(key))?
            .with_dns()?
            .with_relay_client(noise::Config::new, yamux::Config::default)?
            .with_behaviour(|key, relay_client| behaviour(key, Some(relay_client)))?
            .with_swarm_config(config)
            .build(),
        NodeTransports::OnionOnly { socks } => builder
            .with_other_transport(|key| onion_only_transport(key, socks))?
            .with_behaviour(|key| behaviour(key, None))?
            .with_swarm_config(config)
            .build(),
    })
}

/// A guardian's transports: the loopback listener Tor forwards to, and the
/// onion dial transport. Nothing else exists to dial or resolve with.
fn onion_only_transport(
    key: &Keypair,
    socks: SocketAddr,
) -> Result<Boxed<(PeerId, StreamMuxerBox)>, Box<dyn std::error::Error + Send + Sync>> {
    let listener = LoopbackListener(tcp::tokio::Transport::new(
        tcp::Config::default().nodelay(true),
    ))
    .upgrade(upgrade::Version::V1)
    .authenticate(noise::Config::new(key)?)
    .multiplex(yamux::Config::default())
    .map(|(peer, muxer), _| (peer, StreamMuxerBox::new(muxer)))
    .boxed();
    let dialer = OnionDialTransport::new(socks).authenticated(key)?;
    Ok(listener
        .or_transport(dialer)
        .map(|either, _| either.into_inner())
        .boxed())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onion::test_support::{FakeSocks, FakeSocksConfig};
    use futures::StreamExt;
    use libp2p::swarm::{dummy, DialError, SwarmEvent};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";

    /// A plain TCP listener that counts the connections it accepts: the
    /// proof that a dial reached (or never reached) a socket.
    struct CountingListener {
        port: u16,
        accepted: Arc<AtomicUsize>,
    }

    impl CountingListener {
        async fn start() -> Self {
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
            Self { port, accepted }
        }

        fn accepted(&self) -> usize {
            self.accepted.load(Ordering::SeqCst)
        }
    }

    fn quick(config: libp2p::swarm::Config) -> libp2p::swarm::Config {
        config.with_idle_connection_timeout(Duration::from_secs(30))
    }

    fn swarm(transports: NodeTransports) -> Swarm<dummy::Behaviour> {
        build_swarm(
            Keypair::generate_ed25519(),
            transports,
            move |_, relay_client| {
                assert_eq!(
                    relay_client.is_some(),
                    !matches!(transports, NodeTransports::OnionOnly { .. }),
                    "a relay client exists exactly when clearnet does"
                );
                Ok(dummy::Behaviour)
            },
            quick,
        )
        .unwrap()
    }

    /// Dial every address and return each dial's transport errors.
    async fn dial_errors(
        swarm: &mut Swarm<dummy::Behaviour>,
        addrs: &[Multiaddr],
    ) -> Vec<Vec<(Multiaddr, String, bool)>> {
        for addr in addrs {
            swarm.dial(addr.clone()).unwrap();
        }
        let mut errors = Vec::new();
        tokio::time::timeout(Duration::from_secs(20), async {
            while errors.len() < addrs.len() {
                if let SwarmEvent::OutgoingConnectionError { error, .. } =
                    swarm.select_next_some().await
                {
                    let DialError::Transport(failures) = error else {
                        panic!("unexpected dial error: {error}");
                    };
                    errors.push(
                        failures
                            .into_iter()
                            .map(|(addr, error)| {
                                let unsupported =
                                    matches!(error, TransportError::MultiaddrNotSupported(_));
                                (addr, error.to_string(), unsupported)
                            })
                            .collect(),
                    );
                }
            }
        })
        .await
        .expect("every dial fails");
        errors
    }

    #[tokio::test]
    async fn guardian_swarm_never_opens_a_clearnet_socket_nor_resolves_a_name() {
        let target = CountingListener::start().await;
        let proxy =
            FakeSocks::start(FakeSocksConfig::forwarding("127.0.0.1:9".parse().unwrap())).await;
        let port = target.port;
        let (relay, destination) = (PeerId::random(), PeerId::random());
        let clearnet: Vec<Multiaddr> = [
            format!("/ip4/127.0.0.1/tcp/{port}"),
            format!("/ip6/::1/tcp/{port}"),
            format!("/dns4/localhost/tcp/{port}"),
            format!("/dns6/localhost/tcp/{port}"),
            format!("/dns/localhost/tcp/{port}"),
            "/dnsaddr/localhost".to_string(),
            format!("/dns/{SYNTHETIC}.onion/tcp/{port}"),
            format!("/ip4/127.0.0.1/tcp/{port}/p2p/{relay}/p2p-circuit/p2p/{destination}"),
        ]
        .iter()
        .map(|text| text.parse().unwrap())
        .collect();

        let mut guardian = swarm(NodeTransports::OnionOnly { socks: proxy.addr });
        for failures in dial_errors(&mut guardian, &clearnet).await {
            for (addr, error, unsupported) in failures {
                // Refused by every transport before any I/O. A DNS layer
                // would accept a `/dns*` address and fail only after the
                // lookup, with another error.
                assert!(unsupported, "{addr} was attempted: {error}");
            }
        }
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(target.accepted(), 0, "a clearnet socket was opened");
        assert_eq!(proxy.connections(), 0, "only onion addresses go to Tor");

        // Control: the counter does see a clearnet dial, by address and by
        // name, from a clearnet swarm.
        let mut ordinary = swarm(NodeTransports::Clearnet);
        for addr in [&clearnet[0], &clearnet[2]] {
            ordinary.dial(addr.clone()).unwrap();
        }
        tokio::time::timeout(Duration::from_secs(20), async {
            while target.accepted() < 2 {
                tokio::select! {
                    _ = ordinary.select_next_some() => {}
                    _ = tokio::time::sleep(Duration::from_millis(50)) => {}
                }
            }
        })
        .await
        .expect("the control dials reach the counting listener");
    }

    #[tokio::test]
    async fn guardian_swarm_listens_only_on_loopback() {
        let mut guardian = swarm(NodeTransports::OnionOnly {
            socks: "127.0.0.1:9".parse().unwrap(),
        });
        for refused in [
            "/ip4/0.0.0.0/tcp/0",
            "/ip6/::/tcp/0",
            "/ip4/192.0.2.1/tcp/9700",
            "/dns4/localhost/tcp/0",
            "/ip4/127.0.0.1/tcp/0/p2p-circuit",
        ] {
            assert!(
                guardian.listen_on(refused.parse().unwrap()).is_err(),
                "{refused} accepted as a guardian listener"
            );
        }
        guardian
            .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
            .unwrap();
        let bound = tokio::time::timeout(Duration::from_secs(10), async {
            loop {
                if let SwarmEvent::NewListenAddr { address, .. } = guardian.select_next_some().await
                {
                    return address;
                }
            }
        })
        .await
        .unwrap();
        assert!(is_loopback_tcp(&bound), "{bound}");

        let mut listener = LoopbackListener(libp2p::tcp::tokio::Transport::new(
            libp2p::tcp::Config::default(),
        ));
        assert!(matches!(
            listener.dial(
                bound.clone(),
                DialOpts {
                    role: libp2p::core::Endpoint::Dialer,
                    port_use: libp2p::core::transport::PortUse::New,
                },
            ),
            Err(TransportError::MultiaddrNotSupported(_))
        ));
    }

    /// A swarm listening on loopback, and its dialable address.
    async fn listening_peer() -> (Swarm<dummy::Behaviour>, PeerId, u16) {
        let mut peer = swarm(NodeTransports::Clearnet);
        peer.listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
            .unwrap();
        let address = loop {
            if let SwarmEvent::NewListenAddr { address, .. } = peer.select_next_some().await {
                break address;
            }
        };
        let Some(Protocol::Tcp(port)) = address.iter().last() else {
            panic!("{address}");
        };
        let id = *peer.local_peer_id();
        (peer, id, port)
    }

    async fn connect(
        dialer: &mut Swarm<dummy::Behaviour>,
        listener: &mut Swarm<dummy::Behaviour>,
        expected: PeerId,
    ) -> Multiaddr {
        tokio::time::timeout(Duration::from_secs(20), async {
            loop {
                tokio::select! {
                    event = dialer.select_next_some() => match event {
                        SwarmEvent::ConnectionEstablished { peer_id, endpoint, .. } => {
                            assert_eq!(peer_id, expected);
                            return endpoint.get_remote_address().clone();
                        }
                        SwarmEvent::OutgoingConnectionError { error, .. } => {
                            panic!("dial failed: {error}");
                        }
                        _ => {}
                    },
                    _ = listener.select_next_some() => {}
                }
            }
        })
        .await
        .expect("connected within 20 s")
    }

    #[tokio::test]
    async fn guardian_swarm_reaches_an_onion_peer_through_the_socks_proxy() {
        let (mut peer, peer_id, port) = listening_peer().await;
        let proxy = FakeSocks::start(FakeSocksConfig::forwarding(
            format!("127.0.0.1:{port}").parse().unwrap(),
        ))
        .await;
        let mut guardian = swarm(NodeTransports::OnionOnly { socks: proxy.addr });
        let onion: Multiaddr = format!("/onion3/{SYNTHETIC}:9700/p2p/{peer_id}")
            .parse()
            .unwrap();
        guardian.dial(onion).unwrap();
        let remote = connect(&mut guardian, &mut peer, peer_id).await;
        assert!(matches!(remote.iter().next(), Some(Protocol::Onion3(_))));
        let requests = proxy.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].domain().unwrap(), format!("{SYNTHETIC}.onion"));
    }

    #[tokio::test]
    async fn dual_swarm_keeps_clearnet_off_the_proxy_and_onion_on_it() {
        let (mut onion_peer, onion_id, onion_port) = listening_peer().await;
        let proxy = FakeSocks::start(FakeSocksConfig::forwarding(
            format!("127.0.0.1:{onion_port}").parse().unwrap(),
        ))
        .await;
        let mut dual = swarm(NodeTransports::ClearnetAndOnion { socks: proxy.addr });
        dual.dial(
            format!("/onion3/{SYNTHETIC}:9700/p2p/{onion_id}")
                .parse::<Multiaddr>()
                .unwrap(),
        )
        .unwrap();
        connect(&mut dual, &mut onion_peer, onion_id).await;
        assert_eq!(proxy.connections(), 1);

        let (mut clear_peer, clear_id, clear_port) = listening_peer().await;
        dual.dial(
            format!("/ip4/127.0.0.1/tcp/{clear_port}/p2p/{clear_id}")
                .parse::<Multiaddr>()
                .unwrap(),
        )
        .unwrap();
        let remote = connect(&mut dual, &mut clear_peer, clear_id).await;
        assert!(matches!(remote.iter().next(), Some(Protocol::Ip4(_))));
        assert_eq!(proxy.connections(), 1, "a clearnet dial went through Tor");
    }

    #[test]
    fn transports_follow_the_onion_mode() {
        use crate::onion::config::OnionConfig;
        assert_eq!(NodeTransports::of(None), NodeTransports::Clearnet);
        let settings = |mode| {
            OnionConfig {
                enabled: true,
                mode,
                ..OnionConfig::default()
            }
            .validate()
            .unwrap()
            .unwrap()
        };
        let socks: SocketAddr = "127.0.0.1:9050".parse().unwrap();
        assert_eq!(
            NodeTransports::of(Some(&settings(OnionMode::Additional))),
            NodeTransports::ClearnetAndOnion { socks }
        );
        assert_eq!(
            NodeTransports::of(Some(&settings(OnionMode::Guardian))),
            NodeTransports::OnionOnly { socks }
        );
    }
}
