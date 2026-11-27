// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! An outbound-only libp2p transport for `/onion3/` addresses.
//!
//! It dials through Tor's SOCKS port and refuses every other address with
//! `MultiaddrNotSupported`, so composed after TCP (`with_other_transport`)
//! it changes nothing for clearnet dials: TCP never accepts `/onion3/`, this
//! transport never accepts anything else, and the DNS layer has no `/dns`
//! component to resolve in an onion address. It never listens: inbound onion
//! connections arrive on the node's ordinary loopback TCP listener, to which
//! Tor forwards the service.
//!
//! The produced stream is a plain byte pipe; Noise and Yamux run over it
//! exactly as over TCP, so the remote `PeerId` is authenticated end to end
//! whatever the path.

use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use futures::future::BoxFuture;
use libp2p::core::muxing::StreamMuxerBox;
use libp2p::core::transport::{Boxed, DialOpts, ListenerId, TransportError, TransportEvent};
use libp2p::core::{upgrade, Transport};
use libp2p::{identity::Keypair, noise, yamux, Multiaddr, PeerId};

use super::address::parse_onion3_multiaddr;
use super::socks5::{self, SOCKS_CONNECT_TIMEOUT};

/// Dials `/onion3/<id>:<port>[/p2p/<peer>]` through a SOCKS5 proxy.
#[derive(Clone, Debug)]
pub struct OnionDialTransport {
    socks: SocketAddr,
    timeout: Duration,
}

impl OnionDialTransport {
    pub fn new(socks: SocketAddr) -> Self {
        Self {
            socks,
            timeout: SOCKS_CONNECT_TIMEOUT,
        }
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The transport upgraded like the TCP one (Noise, then Yamux), ready
    /// for `SwarmBuilder::with_other_transport` (whose constructor error type
    /// is a boxed error).
    pub fn authenticated(
        self,
        keypair: &Keypair,
    ) -> Result<Boxed<(PeerId, StreamMuxerBox)>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(self
            .upgrade(upgrade::Version::V1)
            .authenticate(noise::Config::new(keypair)?)
            .multiplex(yamux::Config::default())
            .map(|(peer, muxer), _| (peer, StreamMuxerBox::new(muxer)))
            .boxed())
    }
}

impl Transport for OnionDialTransport {
    type Output = libp2p::tcp::tokio::TcpStream;
    type Error = std::io::Error;
    type ListenerUpgrade = futures::future::Pending<Result<Self::Output, Self::Error>>;
    type Dial = BoxFuture<'static, Result<Self::Output, Self::Error>>;

    fn listen_on(
        &mut self,
        _id: ListenerId,
        addr: Multiaddr,
    ) -> Result<(), TransportError<Self::Error>> {
        Err(TransportError::MultiaddrNotSupported(addr))
    }

    fn remove_listener(&mut self, _id: ListenerId) -> bool {
        false
    }

    fn dial(
        &mut self,
        addr: Multiaddr,
        _opts: DialOpts,
    ) -> Result<Self::Dial, TransportError<Self::Error>> {
        let Some((target, _peer)) = parse_onion3_multiaddr(&addr) else {
            return Err(TransportError::MultiaddrNotSupported(addr));
        };
        let socks = self.socks;
        let timeout = self.timeout;
        Ok(Box::pin(async move {
            socks5::connect(socks, &target, timeout)
                .await
                .map(libp2p::tcp::tokio::TcpStream)
                .map_err(std::io::Error::other)
        }))
    }

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<TransportEvent<Self::ListenerUpgrade, Self::Error>> {
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onion::test_support::{echo_server, FakeSocks, FakeSocksConfig};
    use futures::{AsyncReadExt, AsyncWriteExt, StreamExt};
    use libp2p::core::transport::PortUse;
    use libp2p::core::Endpoint;
    use libp2p::multiaddr::Protocol;
    use libp2p::swarm::{dummy, SwarmEvent};

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";

    fn opts() -> DialOpts {
        DialOpts {
            role: Endpoint::Dialer,
            port_use: PortUse::New,
        }
    }

    fn onion_addr() -> Multiaddr {
        format!("/onion3/{SYNTHETIC}:9700").parse().unwrap()
    }

    #[test]
    fn onion_transport_refuses_every_clearnet_address_and_never_listens() {
        let mut transport = OnionDialTransport::new("127.0.0.1:9".parse().unwrap());
        for text in [
            "/ip4/127.0.0.1/tcp/9700".to_string(),
            "/ip6/::1/tcp/9700".to_string(),
            "/dns4/example.org/tcp/9700".to_string(),
            format!("/dns/{SYNTHETIC}.onion/tcp/9700"),
            format!("/onion3/{SYNTHETIC}:9700/tcp/9700"),
            format!(
                "/onion3/{SYNTHETIC}:9700/p2p/{}/p2p-circuit",
                PeerId::random()
            ),
        ] {
            let addr: Multiaddr = text.parse().unwrap();
            assert!(
                matches!(
                    transport.dial(addr.clone(), opts()),
                    Err(TransportError::MultiaddrNotSupported(refused)) if refused == addr
                ),
                "{text} not refused"
            );
        }
        assert!(matches!(
            transport.listen_on(ListenerId::next(), onion_addr()),
            Err(TransportError::MultiaddrNotSupported(_))
        ));
        assert!(matches!(
            transport.listen_on(ListenerId::next(), "/ip4/127.0.0.1/tcp/0".parse().unwrap()),
            Err(TransportError::MultiaddrNotSupported(_))
        ));
    }

    #[tokio::test]
    async fn onion_transport_dials_through_the_socks_proxy_by_name() {
        let echo = echo_server().await;
        let proxy = FakeSocks::start(FakeSocksConfig::forwarding(echo)).await;
        let mut transport = OnionDialTransport::new(proxy.addr);
        let peer = PeerId::random();
        for addr in [onion_addr(), onion_addr().with(Protocol::P2p(peer))] {
            let dial = transport.dial(addr, opts()).unwrap();
            // Nothing happens before the future is polled.
            let mut stream = dial.await.unwrap();
            stream.write_all(b"onion").await.unwrap();
            let mut echoed = [0u8; 5];
            stream.read_exact(&mut echoed).await.unwrap();
            assert_eq!(&echoed, b"onion");
        }
        let requests = proxy.requests();
        assert_eq!(requests.len(), 2);
        for request in requests {
            assert_eq!(request.domain().unwrap(), format!("{SYNTHETIC}.onion"));
            assert_eq!(request.port(), Some(9700));
        }
    }

    #[tokio::test]
    async fn onion_transport_dial_failure_is_an_io_error() {
        let echo = echo_server().await;
        let proxy = FakeSocks::start(FakeSocksConfig {
            connect_reply: vec![5, 0xf6, 0, 1, 0, 0, 0, 0, 0, 0],
            ..FakeSocksConfig::forwarding(echo)
        })
        .await;
        let mut transport = OnionDialTransport::new(proxy.addr);
        assert!(transport.dial(onion_addr(), opts()).unwrap().await.is_err());
    }

    fn swarm_with_onion(socks: SocketAddr) -> libp2p::Swarm<dummy::Behaviour> {
        libp2p::SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                libp2p::tcp::Config::default().nodelay(true),
                noise::Config::new,
                libp2p::yamux::Config::default,
            )
            .unwrap()
            .with_other_transport(|key| {
                OnionDialTransport::new(socks)
                    .with_timeout(Duration::from_secs(10))
                    .authenticated(key)
            })
            .unwrap()
            .with_dns()
            .unwrap()
            .with_behaviour(|_| dummy::Behaviour)
            .unwrap()
            .with_swarm_config(|config| {
                config.with_idle_connection_timeout(Duration::from_secs(30))
            })
            .build()
    }

    fn plain_swarm() -> libp2p::Swarm<dummy::Behaviour> {
        libp2p::SwarmBuilder::with_new_identity()
            .with_tokio()
            .with_tcp(
                libp2p::tcp::Config::default().nodelay(true),
                noise::Config::new,
                libp2p::yamux::Config::default,
            )
            .unwrap()
            .with_behaviour(|_| dummy::Behaviour)
            .unwrap()
            .with_swarm_config(|config| {
                config.with_idle_connection_timeout(Duration::from_secs(30))
            })
            .build()
    }

    async fn listen_loopback(swarm: &mut libp2p::Swarm<dummy::Behaviour>) -> Multiaddr {
        swarm
            .listen_on("/ip4/127.0.0.1/tcp/0".parse().unwrap())
            .unwrap();
        loop {
            if let SwarmEvent::NewListenAddr { address, .. } = swarm.select_next_some().await {
                return address;
            }
        }
    }

    /// Drive both swarms until `dialer` is connected to `expected`; returns
    /// the dialer-side remote address.
    async fn connect_both(
        dialer: &mut libp2p::Swarm<dummy::Behaviour>,
        listener: &mut libp2p::Swarm<dummy::Behaviour>,
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
        .expect("connection within 20 s")
    }

    #[tokio::test]
    async fn onion_transport_composes_beside_tcp_and_dns_in_a_swarm() {
        let mut listener = plain_swarm();
        let listener_peer = *listener.local_peer_id();
        let listen_addr = listen_loopback(&mut listener).await;
        let Some(Protocol::Tcp(port)) = listen_addr.iter().last() else {
            panic!("tcp listen address expected: {listen_addr}");
        };
        let proxy = FakeSocks::start(FakeSocksConfig::forwarding(
            format!("127.0.0.1:{port}").parse().unwrap(),
        ))
        .await;

        // Onion path: through the proxy, by name, authenticated end to end.
        let mut dialer = swarm_with_onion(proxy.addr);
        let onion = onion_addr().with(Protocol::P2p(listener_peer));
        dialer.dial(onion.clone()).unwrap();
        let remote = connect_both(&mut dialer, &mut listener, listener_peer).await;
        assert!(remote
            .iter()
            .any(|protocol| matches!(protocol, Protocol::Onion3(_))));
        let requests = proxy.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].domain().unwrap(), format!("{SYNTHETIC}.onion"));

        // Clearnet path on the same swarm: TCP as before, the proxy untouched.
        let mut second_listener = plain_swarm();
        let second_peer = *second_listener.local_peer_id();
        let second_addr = listen_loopback(&mut second_listener).await;
        dialer
            .dial(second_addr.clone().with(Protocol::P2p(second_peer)))
            .unwrap();
        let remote = connect_both(&mut dialer, &mut second_listener, second_peer).await;
        assert!(!remote
            .iter()
            .any(|protocol| matches!(protocol, Protocol::Onion3(_))));
        assert_eq!(proxy.connections(), 1);
    }
}
