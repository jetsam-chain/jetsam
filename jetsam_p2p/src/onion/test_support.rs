// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Local stand-ins for Tor, used only by the tests: a ControlPort speaking
//! the text protocol, a scripted line server, a SOCKS5 proxy that forwards
//! to a local listener, and an echo server. Everything binds 127.0.0.1:0;
//! nothing here talks to the Tor network or leaves the host.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::watch;

use crate::onion::key_store::tests::{KEY_BLOB, SYNTHETIC};

fn quote(bytes: &[u8]) -> String {
    let mut out = String::from("\"");
    for &byte in bytes {
        match byte {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            b'\n' => out.push_str("\\n"),
            0x20..=0x7e => out.push(byte as char),
            other => out.push_str(&format!("\\{other:03o}")),
        }
    }
    out.push('"');
    out
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(text.get(index..index + 2)?, 16).ok())
        .collect()
}

/// A line server that answers the n-th received line with the n-th scripted
/// reply, closes once the script is exhausted, and never answers when the
/// script is empty (until [`ScriptedServer::close_all`]).
pub(crate) struct ScriptedServer {
    pub(crate) addr: SocketAddr,
    received: Arc<Mutex<Vec<String>>>,
    close: watch::Sender<bool>,
}

impl ScriptedServer {
    pub(crate) async fn start(script: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        let (close, close_rx) = watch::channel(false);
        let log = received.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let script = script.clone();
                let log = log.clone();
                let mut close_rx = close_rx.clone();
                tokio::spawn(async move {
                    let mut stream = BufReader::new(stream);
                    let mut next = 0usize;
                    loop {
                        let mut line = String::new();
                        tokio::select! {
                            read = stream.read_line(&mut line) => {
                                if read.unwrap_or(0) == 0 {
                                    return;
                                }
                            }
                            _ = close_rx.changed() => return,
                        }
                        log.lock().unwrap().push(line);
                        let Some(reply) = script.get(next) else {
                            if script.is_empty() {
                                continue;
                            }
                            return;
                        };
                        next += 1;
                        if stream.get_mut().write_all(reply.as_bytes()).await.is_err() {
                            return;
                        }
                        if next == script.len() {
                            let _ = stream.get_mut().shutdown().await;
                            return;
                        }
                    }
                });
            }
        });
        Self {
            addr,
            received,
            close,
        }
    }

    pub(crate) fn received(&self) -> Vec<String> {
        self.received.lock().unwrap().clone()
    }

    pub(crate) fn close_all(&self) {
        let _ = self.close.send(true);
    }
}

/// How the fake ControlPort behaves.
#[derive(Clone)]
pub(crate) struct FakeTorConfig {
    pub(crate) methods: &'static str,
    pub(crate) cookie_file: Option<PathBuf>,
    pub(crate) cookie: Option<[u8; 32]>,
    pub(crate) password: Option<Vec<u8>>,
    /// Returned for `ADD_ONION NEW:ED25519-V3`.
    pub(crate) new_service_id: String,
    pub(crate) new_key_blob: String,
    /// Returned for `ADD_ONION ED25519-V3:<key>`.
    pub(crate) existing_service_id: String,
    pub(crate) add_onion_error: Option<String>,
}

impl FakeTorConfig {
    pub(crate) fn null() -> Self {
        Self {
            methods: "NULL",
            cookie_file: None,
            cookie: None,
            password: None,
            new_service_id: SYNTHETIC.into(),
            new_key_blob: KEY_BLOB.into(),
            existing_service_id: SYNTHETIC.into(),
            add_onion_error: None,
        }
    }

    pub(crate) fn cookie(path: &Path, cookie: [u8; 32]) -> Self {
        Self {
            methods: "COOKIE,SAFECOOKIE",
            cookie_file: Some(path.to_path_buf()),
            cookie: Some(cookie),
            ..Self::null()
        }
    }

    pub(crate) fn password(password: &[u8]) -> Self {
        Self {
            methods: "HASHEDPASSWORD",
            password: Some(password.to_vec()),
            ..Self::null()
        }
    }
}

/// A ControlPort with Tor's observable behaviour for the commands we use:
/// commands must end with CRLF, everything but PROTOCOLINFO/AUTHENTICATE is
/// refused with 514 before authentication, a failed AUTHENTICATE closes the
/// connection, and a service exists only while its connection is open.
pub(crate) struct FakeTor {
    pub(crate) addr: SocketAddr,
    received: Arc<Mutex<Vec<String>>>,
    connections: Arc<AtomicUsize>,
    close: watch::Sender<bool>,
}

impl FakeTor {
    pub(crate) async fn start(config: FakeTorConfig) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let received = Arc::new(Mutex::new(Vec::new()));
        let connections = Arc::new(AtomicUsize::new(0));
        let (close, close_rx) = watch::channel(false);
        let log = received.clone();
        let count = connections.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                count.fetch_add(1, Ordering::SeqCst);
                let config = config.clone();
                let log = log.clone();
                let mut close_rx = close_rx.clone();
                close_rx.borrow_and_update();
                tokio::spawn(Self::session(stream, config, log, close_rx));
            }
        });
        Self {
            addr,
            received,
            connections,
            close,
        }
    }

    async fn session(
        stream: TcpStream,
        config: FakeTorConfig,
        log: Arc<Mutex<Vec<String>>>,
        mut close_rx: watch::Receiver<bool>,
    ) {
        let mut stream = BufReader::new(stream);
        let mut authenticated = false;
        loop {
            let mut line = String::new();
            tokio::select! {
                read = stream.read_line(&mut line) => {
                    if read.unwrap_or(0) == 0 {
                        return;
                    }
                }
                _ = close_rx.changed() => return,
            }
            log.lock().unwrap().push(line.clone());
            let Some(command) = line.strip_suffix("\r\n") else {
                let _ = stream
                    .get_mut()
                    .write_all(b"500 Line not terminated by CRLF\r\n")
                    .await;
                return;
            };
            let (word, args) = command.split_once(' ').unwrap_or((command, ""));
            let (reply, close_after) = match word.to_ascii_uppercase().as_str() {
                "PROTOCOLINFO" => {
                    let cookie = config
                        .cookie_file
                        .as_ref()
                        .map(|path| {
                            format!(" COOKIEFILE={}", quote(path.as_os_str().as_encoded_bytes()))
                        })
                        .unwrap_or_default();
                    (
                        format!(
                            "250-PROTOCOLINFO 1\r\n250-AUTH METHODS={}{cookie}\r\n250-VERSION Tor=\"0.4.8.12\"\r\n250 OK\r\n",
                            config.methods
                        ),
                        false,
                    )
                }
                "AUTHENTICATE" => {
                    let secret = if args.starts_with('"') {
                        crate::onion::control::unquote(args)
                            .ok()
                            .map(|(value, _)| value)
                    } else {
                        unhex(args)
                    };
                    let accepted = config.methods.contains("NULL")
                        || (config.methods.contains("COOKIE")
                            && config.cookie.is_some_and(|cookie| {
                                secret.as_deref() == Some(cookie.as_slice())
                            }))
                        || (config.methods.contains("HASHEDPASSWORD")
                            && config.password.is_some()
                            && secret == config.password);
                    if accepted {
                        authenticated = true;
                        ("250 OK\r\n".to_string(), false)
                    } else {
                        (
                            "515 Authentication failed: Password did not match HashedControlPassword *or* authentication cookie.\r\n"
                                .to_string(),
                            true,
                        )
                    }
                }
                _ if !authenticated => ("514 Authentication required.\r\n".to_string(), true),
                "ADD_ONION" => match &config.add_onion_error {
                    Some(error) => (format!("{error}\r\n"), false),
                    None => {
                        let (key, port) = args.split_once(' ').unwrap_or((args, ""));
                        let port_ok = port
                            .strip_prefix("Port=")
                            .and_then(|spec| spec.split_once(','))
                            .is_some_and(|(virtual_port, target)| {
                                virtual_port.parse::<u16>().is_ok()
                                    && target.parse::<SocketAddr>().is_ok()
                            });
                        if !port_ok {
                            ("512 Invalid VIRTPORT/TARGET\r\n".to_string(), false)
                        } else if key == "NEW:ED25519-V3" {
                            (
                                format!(
                                    "250-ServiceID={}\r\n250-PrivateKey={}\r\n250 OK\r\n",
                                    config.new_service_id, config.new_key_blob
                                ),
                                false,
                            )
                        } else if key.starts_with("ED25519-V3:") {
                            (
                                format!(
                                    "250-ServiceID={}\r\n250 OK\r\n",
                                    config.existing_service_id
                                ),
                                false,
                            )
                        } else {
                            ("513 Invalid key type\r\n".to_string(), false)
                        }
                    }
                },
                "DEL_ONION" => ("250 OK\r\n".to_string(), false),
                _ => (format!("510 Unrecognized command \"{word}\"\r\n"), false),
            };
            if stream.get_mut().write_all(reply.as_bytes()).await.is_err() || close_after {
                return;
            }
        }
    }

    pub(crate) fn received(&self) -> Vec<String> {
        self.received.lock().unwrap().clone()
    }

    pub(crate) fn connections(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }

    /// Drop every session, as a Tor restart would.
    pub(crate) fn close_all(&self) {
        let _ = self.close.send(true);
    }
}

/// How the fake SOCKS5 proxy answers.
#[derive(Clone)]
pub(crate) struct FakeSocksConfig {
    /// The method-selection reply.
    pub(crate) method_reply: [u8; 2],
    /// The full reply to CONNECT (header and bound address).
    pub(crate) connect_reply: Vec<u8>,
    /// Where a successful CONNECT is forwarded.
    pub(crate) forward_to: Option<SocketAddr>,
    /// Accept, then never answer.
    pub(crate) silent: bool,
}

impl FakeSocksConfig {
    pub(crate) fn forwarding(target: SocketAddr) -> Self {
        Self {
            method_reply: [5, 0],
            connect_reply: vec![5, 0, 0, 1, 127, 0, 0, 1, 0, 0],
            forward_to: Some(target),
            silent: false,
        }
    }
}

/// What the proxy received on one connection.
#[derive(Clone, Debug, Default)]
pub(crate) struct SocksRequest {
    pub(crate) greeting: Vec<u8>,
    pub(crate) request: Vec<u8>,
}

impl SocksRequest {
    /// The domain of an ATYP=3 request.
    pub(crate) fn domain(&self) -> Option<String> {
        if self.request.get(3) != Some(&3) {
            return None;
        }
        let len = usize::from(*self.request.get(4)?);
        String::from_utf8(self.request.get(5..5 + len)?.to_vec()).ok()
    }

    pub(crate) fn port(&self) -> Option<u16> {
        let len = self.request.len();
        Some(u16::from_be_bytes([
            *self.request.get(len.checked_sub(2)?)?,
            *self.request.get(len - 1)?,
        ]))
    }
}

pub(crate) struct FakeSocks {
    pub(crate) addr: SocketAddr,
    requests: Arc<Mutex<Vec<SocksRequest>>>,
    connections: Arc<AtomicUsize>,
}

impl FakeSocks {
    pub(crate) async fn start(config: FakeSocksConfig) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let connections = Arc::new(AtomicUsize::new(0));
        let log = requests.clone();
        let count = connections.clone();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                count.fetch_add(1, Ordering::SeqCst);
                let config = config.clone();
                let log = log.clone();
                tokio::spawn(async move {
                    let _ = Self::session(stream, config, log).await;
                });
            }
        });
        Self {
            addr,
            requests,
            connections,
        }
    }

    async fn session(
        mut stream: TcpStream,
        config: FakeSocksConfig,
        log: Arc<Mutex<Vec<SocksRequest>>>,
    ) -> std::io::Result<()> {
        if config.silent {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            return Ok(());
        }
        let index = {
            let mut log = log.lock().unwrap();
            log.push(SocksRequest::default());
            log.len() - 1
        };
        let mut head = [0u8; 2];
        stream.read_exact(&mut head).await?;
        let mut methods = vec![0u8; usize::from(head[1])];
        stream.read_exact(&mut methods).await?;
        let mut greeting = head.to_vec();
        greeting.extend_from_slice(&methods);
        log.lock().unwrap()[index].greeting = greeting;
        stream.write_all(&config.method_reply).await?;
        if config.method_reply != [5, 0] {
            return Ok(());
        }
        let mut request = vec![0u8; 4];
        stream.read_exact(&mut request).await?;
        let address_len = match request[3] {
            1 => 4,
            4 => 16,
            3 => {
                let mut len = [0u8; 1];
                stream.read_exact(&mut len).await?;
                request.push(len[0]);
                usize::from(len[0])
            }
            _ => return Ok(()),
        };
        let mut rest = vec![0u8; address_len + 2];
        stream.read_exact(&mut rest).await?;
        request.extend_from_slice(&rest);
        log.lock().unwrap()[index].request = request;
        stream.write_all(&config.connect_reply).await?;
        if config.connect_reply.get(1) != Some(&0) {
            return Ok(());
        }
        if let Some(target) = config.forward_to {
            let mut upstream = TcpStream::connect(target).await?;
            tokio::io::copy_bidirectional(&mut stream, &mut upstream).await?;
        }
        Ok(())
    }

    pub(crate) fn requests(&self) -> Vec<SocksRequest> {
        self.requests.lock().unwrap().clone()
    }

    pub(crate) fn connections(&self) -> usize {
        self.connections.load(Ordering::SeqCst)
    }
}

/// An echo server on 127.0.0.1.
pub(crate) async fn echo_server() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            tokio::spawn(async move {
                let (mut reader, mut writer) = stream.split();
                let _ = tokio::io::copy(&mut reader, &mut writer).await;
            });
        }
    });
    addr
}
