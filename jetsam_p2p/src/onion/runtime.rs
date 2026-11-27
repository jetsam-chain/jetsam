// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The onion service of a running node: its loopback listener, its
//! publication through the ControlPort, and the external address it adds.
//!
//! The service is not detached from the control session, so it disappears
//! when Tor restarts. The publication task then reports the withdrawal and
//! publishes again with the stored key: same key, same address. Until it is
//! back, the address is no longer announced.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use libp2p::core::transport::ListenerId;
use libp2p::multiaddr::Protocol;
use libp2p::swarm::{NetworkBehaviour, SwarmEvent};
use libp2p::{Multiaddr, Swarm};
use tokio::sync::mpsc;

use super::announce::addresses_to_announce;
use super::config::{OnionMode, OnionSettings};
use super::service;

/// First delay before publishing again, doubled up to [`MAX_REPUBLISH_DELAY`].
pub const INITIAL_REPUBLISH_DELAY: Duration = Duration::from_secs(5);
pub const MAX_REPUBLISH_DELAY: Duration = Duration::from_secs(5 * 60);

/// What the publication task reports to the swarm task.
#[derive(Debug)]
pub enum OnionUpdate {
    Published(Multiaddr),
    Withdrawn { addr: Multiaddr, reason: String },
    Failed(String),
}

pub struct OnionRuntime {
    settings: OnionSettings,
    key_dir: PathBuf,
    listener: ListenerId,
    published: Option<Multiaddr>,
    updates_tx: mpsc::Sender<OnionUpdate>,
    updates_rx: mpsc::Receiver<OnionUpdate>,
    retry: (Duration, Duration),
    task: Option<tokio::task::JoinHandle<()>>,
}

impl OnionRuntime {
    /// Bind the loopback listener Tor forwards to. `None` when the onion
    /// service is disabled. A guardian that cannot bind it does not start;
    /// a dual node logs the failure and runs on clearnet as before.
    pub fn listen<B: NetworkBehaviour>(
        swarm: &mut Swarm<B>,
        settings: Option<OnionSettings>,
        key_dir: PathBuf,
    ) -> anyhow::Result<Option<Self>> {
        let Some(settings) = settings else {
            return Ok(None);
        };
        let addr = settings.local_listen_multiaddr();
        let listener = match swarm.listen_on(addr.clone()) {
            Ok(listener) => listener,
            Err(error) if settings.mode == OnionMode::Guardian => anyhow::bail!(
                "guardian mode: cannot listen on the onion service's loopback target {addr}: {error}"
            ),
            Err(error) => {
                tracing::warn!(
                    %addr,
                    %error,
                    "onion service off: its loopback target cannot be bound; clearnet only"
                );
                return Ok(None);
            }
        };
        let (updates_tx, updates_rx) = mpsc::channel(8);
        Ok(Some(Self {
            settings,
            key_dir,
            listener,
            published: None,
            updates_tx,
            updates_rx,
            retry: (INITIAL_REPUBLISH_DELAY, MAX_REPUBLISH_DELAY),
            task: None,
        }))
    }

    /// Shorter delays, for tests.
    pub fn with_retry(mut self, initial: Duration, max: Duration) -> Self {
        self.retry = (initial, max);
        self
    }

    pub fn published(&self) -> Option<&Multiaddr> {
        self.published.as_ref()
    }

    /// Look at a swarm event. When the onion listener has just bound, start
    /// publishing and return its local address (inbound connections on it
    /// are onion connections); when it closes, withdraw the service.
    pub fn observe<B: NetworkBehaviour>(
        &mut self,
        swarm: &mut Swarm<B>,
        event: &SwarmEvent<B::ToSwarm>,
    ) -> Option<Multiaddr> {
        match event {
            SwarmEvent::NewListenAddr {
                listener_id,
                address,
            } if *listener_id == self.listener => {
                let target = loopback_socket(address)?;
                tracing::info!(%address, "onion service loopback target listening");
                if self.task.is_none() {
                    self.task = Some(tokio::spawn(keep_published(
                        self.settings.clone(),
                        self.key_dir.clone(),
                        target,
                        self.updates_tx.clone(),
                        self.retry,
                    )));
                }
                Some(address.clone())
            }
            SwarmEvent::ListenerClosed {
                listener_id,
                reason,
                ..
            } if *listener_id == self.listener => {
                if let Some(task) = self.task.take() {
                    task.abort();
                }
                if let Some(addr) = self.published.take() {
                    swarm.remove_external_address(&addr);
                }
                tracing::error!(
                    ?reason,
                    "the onion service's loopback listener closed; the onion service is withdrawn"
                );
                None
            }
            _ => None,
        }
    }

    /// The next report of the publication task; never resolves without a
    /// runtime.
    pub async fn next_update(this: Option<&mut Self>) -> OnionUpdate {
        match this {
            Some(runtime) => match runtime.updates_rx.recv().await {
                Some(update) => update,
                None => std::future::pending().await,
            },
            None => std::future::pending().await,
        }
    }

    /// Register or withdraw the onion address.
    pub fn apply<B: NetworkBehaviour>(&mut self, swarm: &mut Swarm<B>, update: OnionUpdate) {
        let guardian = self.settings.mode == OnionMode::Guardian;
        match update {
            OnionUpdate::Published(addr) => {
                for announced in addresses_to_announce(self.settings.mode, &[], Some(&addr)) {
                    swarm.add_external_address(announced);
                }
                if guardian {
                    tracing::info!(
                        %addr,
                        "onion service published; guardian mode: the only address this node announces"
                    );
                } else {
                    tracing::info!(
                        %addr,
                        "onion service published beside clearnet; dual mode does not hide this node's IP address"
                    );
                }
                self.published = Some(addr);
            }
            OnionUpdate::Withdrawn { addr, reason } => {
                swarm.remove_external_address(&addr);
                if self.published.as_ref() == Some(&addr) {
                    self.published = None;
                }
                tracing::warn!(
                    %addr,
                    %reason,
                    "Tor closed the control session: onion address withdrawn, publishing it again"
                );
            }
            OnionUpdate::Failed(reason) if guardian => tracing::error!(
                %reason,
                "could not publish the onion service; this guardian is unreachable until it is; retrying"
            ),
            OnionUpdate::Failed(reason) => tracing::warn!(
                %reason,
                "could not publish the onion service; retrying (clearnet is unaffected)"
            ),
        }
    }
}

impl Drop for OnionRuntime {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

fn loopback_socket(addr: &Multiaddr) -> Option<SocketAddr> {
    let mut protocols = addr.iter();
    let ip = match protocols.next()? {
        Protocol::Ip4(ip) => std::net::IpAddr::V4(ip),
        Protocol::Ip6(ip) => std::net::IpAddr::V6(ip),
        _ => return None,
    };
    let Some(Protocol::Tcp(port)) = protocols.next() else {
        return None;
    };
    (ip.is_loopback() && port != 0 && protocols.next().is_none()).then(|| SocketAddr::new(ip, port))
}

async fn keep_published(
    settings: OnionSettings,
    key_dir: PathBuf,
    target: SocketAddr,
    updates: mpsc::Sender<OnionUpdate>,
    (initial, max): (Duration, Duration),
) {
    let mut delay = initial;
    loop {
        match service::publish(&settings, &key_dir, target).await {
            Ok(published) => {
                let addr = published.external_addr().clone();
                if updates
                    .send(OnionUpdate::Published(addr.clone()))
                    .await
                    .is_err()
                {
                    return;
                }
                delay = initial;
                // The service lives as long as this session (not detached).
                let reason = published.closed().await.to_string();
                if updates
                    .send(OnionUpdate::Withdrawn { addr, reason })
                    .await
                    .is_err()
                {
                    return;
                }
            }
            Err(error) => {
                if updates
                    .send(OnionUpdate::Failed(error.to_string()))
                    .await
                    .is_err()
                {
                    return;
                }
            }
        }
        tokio::time::sleep(delay).await;
        delay = delay.saturating_mul(2).min(max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onion::config::OnionConfig;
    use crate::onion::key_store::tests::{KEY_BLOB, SYNTHETIC};
    use crate::onion::swarm::{build_swarm, NodeTransports};
    use crate::onion::test_support::{FakeTor, FakeTorConfig};
    use futures::StreamExt;
    use libp2p::identity::Keypair;
    use libp2p::swarm::dummy;

    fn settings(mode: OnionMode, control: SocketAddr, local_port: u16) -> OnionSettings {
        settings_with(mode, control, local_port, None)
    }

    /// C-3: a guardian authenticates with the cookie its bench's Tor names
    /// (`test_support::guardian_tor`).
    fn settings_with(
        mode: OnionMode,
        control: SocketAddr,
        local_port: u16,
        cookie_file: Option<PathBuf>,
    ) -> OnionSettings {
        OnionConfig {
            enabled: true,
            mode,
            control: control.to_string(),
            local_port,
            cookie_file,
            ..OnionConfig::default()
        }
        .validate()
        .unwrap()
        .unwrap()
    }

    fn swarm(settings: &OnionSettings) -> Swarm<dummy::Behaviour> {
        build_swarm(
            Keypair::generate_ed25519(),
            NodeTransports::of(Some(settings)),
            |_, _| Ok(dummy::Behaviour),
            |config| config.with_idle_connection_timeout(Duration::from_secs(30)),
        )
        .unwrap()
    }

    fn onion() -> Multiaddr {
        format!("/onion3/{SYNTHETIC}:9700").parse().unwrap()
    }

    /// Drive the swarm and the runtime until `done` holds; returns the kinds
    /// of the updates applied, in order, and the onion listener's address.
    async fn drive(
        swarm: &mut Swarm<dummy::Behaviour>,
        runtime: &mut OnionRuntime,
        bound: &mut Option<Multiaddr>,
        updates: &mut Vec<&'static str>,
        done: impl Fn(&Swarm<dummy::Behaviour>, &[&'static str]) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(20), async {
            while !done(swarm, updates) {
                tokio::select! {
                    event = swarm.select_next_some() => {
                        if let Some(address) = runtime.observe(swarm, &event) {
                            *bound = Some(address);
                        }
                    }
                    update = OnionRuntime::next_update(Some(&mut *runtime)) => {
                        updates.push(match update {
                            OnionUpdate::Published(_) => "published",
                            OnionUpdate::Withdrawn { .. } => "withdrawn",
                            OnionUpdate::Failed(_) => "failed",
                        });
                        runtime.apply(swarm, update);
                    }
                }
            }
        })
        .await
        .expect("condition reached within 20 s")
    }

    fn add_onion_lines(tor: &FakeTor) -> Vec<String> {
        tor.received()
            .into_iter()
            .filter(|line| line.starts_with("ADD_ONION"))
            .collect()
    }

    #[tokio::test]
    async fn onion_runtime_announces_the_service_and_republishes_it_when_tor_restarts() {
        let directory = tempfile::tempdir().unwrap();
        let cookie_dir = tempfile::tempdir().unwrap();
        let (tor_config, cookie) = crate::onion::test_support::guardian_tor(cookie_dir.path());
        let tor = FakeTor::start(tor_config).await;
        let settings = settings_with(OnionMode::Guardian, tor.addr, 0, Some(cookie));
        let mut swarm = swarm(&settings);
        let mut runtime =
            OnionRuntime::listen(&mut swarm, Some(settings), directory.path().to_path_buf())
                .unwrap()
                .expect("the onion service is enabled")
                .with_retry(Duration::from_millis(50), Duration::from_millis(200));
        let (mut bound, mut updates) = (None, Vec::new());
        drive(
            &mut swarm,
            &mut runtime,
            &mut bound,
            &mut updates,
            |swarm, _| swarm.external_addresses().any(|addr| *addr == onion()),
        )
        .await;
        assert_eq!(
            swarm.external_addresses().cloned().collect::<Vec<_>>(),
            vec![onion()]
        );
        assert_eq!(runtime.published(), Some(&onion()));
        let bound = bound.expect("the onion listener reported its address");
        let port = loopback_socket(&bound)
            .expect("a bound loopback listener")
            .port();

        // Tor restarts: the service is gone with the session, the address is
        // withdrawn, then published again from the stored key.
        tor.close_all();
        drive(
            &mut swarm,
            &mut runtime,
            &mut None,
            &mut updates,
            |swarm, updates| {
                updates.ends_with(&["withdrawn", "published"])
                    && swarm.external_addresses().any(|addr| *addr == onion())
            },
        )
        .await;
        assert_eq!(updates, ["published", "withdrawn", "published"]);
        assert_eq!(
            swarm.external_addresses().cloned().collect::<Vec<_>>(),
            vec![onion()]
        );
        assert_eq!(
            add_onion_lines(&tor),
            vec![
                format!("ADD_ONION NEW:ED25519-V3 Port=9700,127.0.0.1:{port}\r\n"),
                format!("ADD_ONION {KEY_BLOB} Port=9700,127.0.0.1:{port}\r\n"),
            ],
            "same key, same address, same loopback target"
        );
    }

    #[tokio::test]
    async fn onion_runtime_never_announces_a_service_tor_refused_and_keeps_trying() {
        let directory = tempfile::tempdir().unwrap();
        let cookie_dir = tempfile::tempdir().unwrap();
        let (tor_config, cookie) = crate::onion::test_support::guardian_tor(cookie_dir.path());
        let tor = FakeTor::start(FakeTorConfig {
            add_onion_error: Some("551 Failed to add onion service".into()),
            ..tor_config
        })
        .await;
        let settings = settings_with(OnionMode::Guardian, tor.addr, 0, Some(cookie));
        let mut swarm = swarm(&settings);
        let mut runtime =
            OnionRuntime::listen(&mut swarm, Some(settings), directory.path().to_path_buf())
                .unwrap()
                .unwrap()
                .with_retry(Duration::from_millis(50), Duration::from_millis(100));
        let (mut bound, mut updates) = (None, Vec::new());
        drive(
            &mut swarm,
            &mut runtime,
            &mut bound,
            &mut updates,
            |_, updates| updates.len() >= 3,
        )
        .await;
        assert!(updates.iter().all(|kind| *kind == "failed"), "{updates:?}");
        assert!(add_onion_lines(&tor).len() >= 3);
        assert_eq!(swarm.external_addresses().count(), 0);
        assert_eq!(runtime.published(), None);
    }

    #[tokio::test]
    async fn dual_runtime_adds_the_onion_address_beside_the_clearnet_ones() {
        let directory = tempfile::tempdir().unwrap();
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        let settings = settings(OnionMode::Additional, tor.addr, 0);
        let mut swarm = swarm(&settings);
        let public: Multiaddr = "/ip4/8.8.8.8/tcp/9700".parse().unwrap();
        swarm.add_external_address(public.clone());
        let mut runtime =
            OnionRuntime::listen(&mut swarm, Some(settings), directory.path().to_path_buf())
                .unwrap()
                .unwrap();
        let (mut bound, mut updates) = (None, Vec::new());
        drive(
            &mut swarm,
            &mut runtime,
            &mut bound,
            &mut updates,
            |swarm, _| swarm.external_addresses().any(|addr| *addr == onion()),
        )
        .await;
        let mut announced = swarm.external_addresses().cloned().collect::<Vec<_>>();
        announced.sort_by_key(|addr| addr.to_string());
        assert_eq!(announced, vec![public, onion()]);
    }

    #[tokio::test]
    async fn an_unbindable_onion_target_stops_a_guardian_and_not_a_dual_node() {
        let taken = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = taken.local_addr().unwrap().port();
        let directory = tempfile::tempdir().unwrap();
        let control: SocketAddr = "127.0.0.1:9".parse().unwrap();

        let guardian = settings(OnionMode::Guardian, control, port);
        let mut swarm_g = swarm(&guardian);
        assert!(
            OnionRuntime::listen(&mut swarm_g, Some(guardian), directory.path().to_path_buf())
                .is_err()
        );

        let dual = settings(OnionMode::Additional, control, port);
        let mut swarm_d = swarm(&dual);
        assert!(
            OnionRuntime::listen(&mut swarm_d, Some(dual), directory.path().to_path_buf())
                .unwrap()
                .is_none()
        );
        assert!(
            OnionRuntime::listen(&mut swarm_d, None, directory.path().to_path_buf())
                .unwrap()
                .is_none()
        );
    }
}
