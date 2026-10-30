// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The SOCKS5 client (RFC 1928) used to reach onion services through Tor.
//!
//! Only `CONNECT` with the domain-name address type is ever sent, and only
//! for a checksum-validated `<id>.onion` name: Tor resolves the service, this
//! side never performs a DNS lookup, and no IP address is ever handed to the
//! proxy. Only the no-authentication method is offered. Every reply field is
//! checked, and the whole exchange is bounded by one deadline, which covers
//! Tor building the rendezvous circuit before it answers.

use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpStream;

use super::address::OnionTarget;
use super::OnionError;

/// Default bound on reaching an onion service (circuit building included).
pub const SOCKS_CONNECT_TIMEOUT: Duration = Duration::from_secs(60);

/// What a non-zero reply code means, RFC 1928 and Tor's extended codes.
pub fn reply_message(code: u8) -> &'static str {
    match code {
        0x00 => "succeeded",
        0x01 => "general SOCKS server failure",
        0x02 => "connection not allowed by ruleset",
        0x03 => "network unreachable",
        0x04 => "host unreachable",
        0x05 => "connection refused",
        0x06 => "TTL expired",
        0x07 => "command not supported",
        0x08 => "address type not supported",
        0xf0 => "onion service descriptor can not be found",
        0xf1 => "onion service descriptor is invalid",
        0xf2 => "onion service introduction failed",
        0xf3 => "onion service rendezvous failed",
        0xf4 => "onion service missing client authorization",
        0xf5 => "onion service wrong client authorization",
        0xf6 => "onion service invalid address",
        0xf7 => "onion service introduction timed out",
        _ => "unknown SOCKS reply",
    }
}

const VERSION: u8 = 0x05;
const METHOD_NO_AUTH: u8 = 0x00;
const COMMAND_CONNECT: u8 = 0x01;
const RESERVED: u8 = 0x00;
const ATYP_IPV4: u8 = 0x01;
const ATYP_DOMAIN: u8 = 0x03;
const ATYP_IPV6: u8 = 0x04;

fn socks_protocol(message: impl Into<String>) -> OnionError {
    OnionError::SocksProtocol(message.into())
}

/// Run the method negotiation and `CONNECT <id>.onion:<port>` on `stream`.
pub async fn handshake<S>(stream: &mut S, target: &OnionTarget) -> Result<(), OnionError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    stream.write_all(&[VERSION, 1, METHOD_NO_AUTH]).await?;
    let mut choice = [0u8; 2];
    stream.read_exact(&mut choice).await?;
    if choice[0] != VERSION {
        return Err(socks_protocol(format!(
            "proxy answered version {} to the greeting",
            choice[0]
        )));
    }
    if choice[1] != METHOD_NO_AUTH {
        return Err(socks_protocol(format!(
            "proxy requires method {:#04x}; only no-authentication is offered",
            choice[1]
        )));
    }

    // The checksum-validated `<56 symbols>.onion` name: 62 ASCII bytes, so
    // the one-byte length can never truncate it.
    let host = target.service.hostname();
    let host_len = u8::try_from(host.len()).map_err(|_| socks_protocol("host name too long"))?;
    let mut request = Vec::with_capacity(7 + host.len());
    request.extend_from_slice(&[VERSION, COMMAND_CONNECT, RESERVED, ATYP_DOMAIN, host_len]);
    request.extend_from_slice(host.as_bytes());
    request.extend_from_slice(&target.port.to_be_bytes());
    stream.write_all(&request).await?;
    stream.flush().await?;

    let mut head = [0u8; 4];
    stream.read_exact(&mut head).await?;
    if head[0] != VERSION {
        return Err(socks_protocol(format!(
            "proxy answered version {} to CONNECT",
            head[0]
        )));
    }
    if head[1] != 0x00 {
        return Err(OnionError::SocksRefused {
            code: head[1],
            message: reply_message(head[1]),
        });
    }
    if head[2] != RESERVED {
        return Err(socks_protocol(
            "non-zero reserved byte in the CONNECT reply",
        ));
    }
    // The bound address must be consumed whole, or its bytes would be read
    // as the first bytes of the tunnel.
    let bound_len = match head[3] {
        ATYP_IPV4 => 4,
        ATYP_IPV6 => 16,
        ATYP_DOMAIN => {
            let mut len = [0u8; 1];
            stream.read_exact(&mut len).await?;
            if len[0] == 0 {
                return Err(socks_protocol("empty bound domain in the CONNECT reply"));
            }
            usize::from(len[0])
        }
        other => {
            return Err(socks_protocol(format!(
                "unknown bound address type {other:#04x}"
            )))
        }
    };
    let mut bound = vec![0u8; bound_len + 2];
    stream.read_exact(&mut bound).await?;
    Ok(())
}

/// Open a tunnel to `target` through the loopback SOCKS port `proxy`.
pub async fn connect(
    proxy: SocketAddr,
    target: &OnionTarget,
    timeout: Duration,
) -> Result<TcpStream, OnionError> {
    if !proxy.ip().is_loopback() {
        return Err(OnionError::Config(format!(
            "refusing the non-loopback SOCKS proxy {proxy}"
        )));
    }
    tokio::time::timeout(timeout, async {
        let mut stream = TcpStream::connect(proxy).await?;
        stream.set_nodelay(true)?;
        handshake(&mut stream, target).await?;
        Ok(stream)
    })
    .await
    .map_err(|_| OnionError::Timeout("reaching an onion service through SOCKS5"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onion::address::OnionServiceId;
    use crate::onion::test_support::{echo_server, FakeSocks, FakeSocksConfig};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const SYNTHETIC: &str = "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3kead";

    fn target() -> OnionTarget {
        OnionTarget {
            service: OnionServiceId::parse(SYNTHETIC).unwrap(),
            port: 9700,
        }
    }

    async fn round_trip(stream: &mut TcpStream) {
        stream.write_all(b"ping through tor").await.unwrap();
        let mut echoed = [0u8; 16];
        stream.read_exact(&mut echoed).await.unwrap();
        assert_eq!(&echoed, b"ping through tor");
    }

    #[tokio::test]
    async fn socks_connect_sends_the_onion_name_never_an_address() {
        let echo = echo_server().await;
        let proxy = FakeSocks::start(FakeSocksConfig::forwarding(echo)).await;
        let mut stream = connect(proxy.addr, &target(), Duration::from_secs(5))
            .await
            .unwrap();
        round_trip(&mut stream).await;
        let requests = proxy.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].greeting, vec![5, 1, 0]);
        let mut expected = vec![5, 1, 0, 3, 62];
        expected.extend_from_slice(format!("{SYNTHETIC}.onion").as_bytes());
        expected.extend_from_slice(&9700u16.to_be_bytes());
        assert_eq!(requests[0].request, expected);
        assert_eq!(requests[0].domain().unwrap(), format!("{SYNTHETIC}.onion"));
        assert_eq!(requests[0].port(), Some(9700));
    }

    #[tokio::test]
    async fn socks_bound_addresses_of_every_type_are_consumed() {
        let echo = echo_server().await;
        let mut domain_reply = vec![5, 0, 0, 3, 9];
        domain_reply.extend_from_slice(b"bound.tor");
        domain_reply.extend_from_slice(&[0, 80]);
        let mut v6_reply = vec![5, 0, 0, 4];
        v6_reply.extend_from_slice(&[0u8; 16]);
        v6_reply.extend_from_slice(&[0, 80]);
        for reply in [vec![5, 0, 0, 1, 10, 0, 0, 1, 0, 80], domain_reply, v6_reply] {
            let proxy = FakeSocks::start(FakeSocksConfig {
                connect_reply: reply.clone(),
                ..FakeSocksConfig::forwarding(echo)
            })
            .await;
            let mut stream = connect(proxy.addr, &target(), Duration::from_secs(5))
                .await
                .unwrap();
            // A bound address left unread would corrupt the tunnel.
            round_trip(&mut stream).await;
        }
    }

    #[tokio::test]
    async fn socks_failure_replies_are_errors_with_their_code() {
        let echo = echo_server().await;
        for code in [0x01u8, 0x02, 0x04, 0x05, 0x06, 0xf0, 0xf2, 0xf6, 0x7f] {
            let proxy = FakeSocks::start(FakeSocksConfig {
                connect_reply: vec![5, code, 0, 1, 0, 0, 0, 0, 0, 0],
                ..FakeSocksConfig::forwarding(echo)
            })
            .await;
            match connect(proxy.addr, &target(), Duration::from_secs(5)).await {
                Err(OnionError::SocksRefused { code: got, message }) => {
                    assert_eq!(got, code);
                    assert!(!message.is_empty());
                }
                other => panic!("code {code:#04x}: {other:?}"),
            }
        }
        assert_eq!(reply_message(0x04), "host unreachable");
        assert_eq!(
            reply_message(0xf0),
            "onion service descriptor can not be found"
        );
        assert_eq!(reply_message(0x7f), "unknown SOCKS reply");
    }

    #[tokio::test]
    async fn socks_malformed_replies_are_refused() {
        let echo = echo_server().await;
        let method_replies = [[5u8, 0xff], [5, 2], [4, 0]];
        for method_reply in method_replies {
            let proxy = FakeSocks::start(FakeSocksConfig {
                method_reply,
                ..FakeSocksConfig::forwarding(echo)
            })
            .await;
            assert!(
                connect(proxy.addr, &target(), Duration::from_secs(5))
                    .await
                    .is_err(),
                "{method_reply:?} accepted"
            );
        }
        let connect_replies = [
            vec![4, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            vec![5, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            vec![5, 0, 0, 2, 0, 0, 0, 0, 0, 0],
            vec![5, 0, 0, 3, 0, 0, 80],
            vec![5, 0, 0, 1, 0, 0],
        ];
        for connect_reply in connect_replies {
            let proxy = FakeSocks::start(FakeSocksConfig {
                connect_reply: connect_reply.clone(),
                forward_to: None,
                ..FakeSocksConfig::forwarding(echo)
            })
            .await;
            assert!(
                connect(proxy.addr, &target(), Duration::from_secs(5))
                    .await
                    .is_err(),
                "{connect_reply:?} accepted"
            );
        }
    }

    #[tokio::test]
    async fn socks_connect_is_bounded_in_time() {
        let echo = echo_server().await;
        let proxy = FakeSocks::start(FakeSocksConfig {
            silent: true,
            ..FakeSocksConfig::forwarding(echo)
        })
        .await;
        let started = std::time::Instant::now();
        assert!(matches!(
            connect(proxy.addr, &target(), Duration::from_millis(300)).await,
            Err(OnionError::Timeout(_))
        ));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn socks_proxy_must_be_loopback() {
        assert!(matches!(
            connect(
                "192.0.2.1:9050".parse().unwrap(),
                &target(),
                Duration::from_secs(1)
            )
            .await,
            Err(OnionError::Config(_))
        ));
    }
}
