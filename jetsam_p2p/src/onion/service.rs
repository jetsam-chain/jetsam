// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Publishing the node's onion service.
//!
//! `publish` authenticates to the ControlPort, publishes the stored key (or
//! has Tor generate one and stores it before announcing anything), and keeps
//! the control session: the service is not detached, so it disappears with
//! the session — when the node stops, or when Tor restarts. The caller
//! watches [`PublishedOnion::closed`] and publishes again.

use std::net::SocketAddr;
use std::path::Path;

use libp2p::Multiaddr;

use super::address::OnionServiceId;
use super::config::OnionSettings;
use super::control::{choose_auth_for, KeyRequest, TorControl};
use super::key_store;
use super::OnionError;

/// A published service and the control session that keeps it alive.
pub struct PublishedOnion {
    control: TorControl,
    service_id: OnionServiceId,
    external: Multiaddr,
}

impl PublishedOnion {
    pub fn service_id(&self) -> &OnionServiceId {
        &self.service_id
    }

    /// `/onion3/<id>:<virtual_port>`, the address to announce.
    pub fn external_addr(&self) -> &Multiaddr {
        &self.external
    }

    /// Resolves when Tor closes the session; the service is gone by then.
    pub async fn closed(mut self) -> OnionError {
        self.control.wait_closed().await
    }

    /// Remove the service explicitly (`DEL_ONION`).
    pub async fn withdraw(mut self) -> Result<(), OnionError> {
        self.control.del_onion(&self.service_id).await
    }
}

/// Publish the service forwarding `settings.virtual_port` to
/// `local_target`, the node's bound loopback listener.
pub async fn publish(
    settings: &OnionSettings,
    data_dir: &Path,
    local_target: SocketAddr,
) -> Result<PublishedOnion, OnionError> {
    if !local_target.ip().is_loopback() || local_target.port() == 0 {
        return Err(OnionError::Config(format!(
            "the onion service must forward to a bound loopback listener, not {local_target}"
        )));
    }
    // Before touching Tor: a corrupt or unsafe key file stops here, and is
    // never replaced by a new key (which would change the address).
    let stored = key_store::load(data_dir)?;

    let mut control = TorControl::connect(settings.control).await?;
    let info = control.protocol_info().await?;
    let choice = choose_auth_for(&info, &settings.auth, settings.mode)?;
    control.authenticate(&choice).await?;

    let request = match &stored {
        Some(key) => KeyRequest::Existing(key),
        None => KeyRequest::New,
    };
    let added = control
        .add_onion(request, settings.virtual_port, local_target)
        .await?;
    if let Some(new_key) = &added.new_key {
        // An address that cannot be kept is not announced: withdraw it.
        if let Err(error) = key_store::store_new(data_dir, new_key) {
            let _ = control.del_onion(&added.service_id).await;
            return Err(error);
        }
    }
    let external = added.service_id.to_multiaddr(settings.virtual_port)?;
    Ok(PublishedOnion {
        control,
        service_id: added.service_id,
        external,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onion::config::OnionConfig;
    use crate::onion::key_store::{self, tests::KEY_BLOB, tests::SYNTHETIC, ONION_KEY_FILE};
    use crate::onion::test_support::{FakeTor, FakeTorConfig};
    use std::time::Duration;

    fn settings(control: SocketAddr) -> OnionSettings {
        OnionConfig {
            enabled: true,
            control: control.to_string(),
            ..OnionConfig::default()
        }
        .validate()
        .unwrap()
        .unwrap()
    }

    fn target() -> SocketAddr {
        "127.0.0.1:19700".parse().unwrap()
    }

    fn add_onion_lines(tor: &FakeTor) -> Vec<String> {
        tor.received()
            .into_iter()
            .filter(|line| line.starts_with("ADD_ONION"))
            .collect()
    }

    #[tokio::test]
    async fn onion_publish_creates_stores_and_then_reuses_the_key() {
        let directory = tempfile::tempdir().unwrap();
        let tor = FakeTor::start(FakeTorConfig::null()).await;

        let first = publish(&settings(tor.addr), directory.path(), target())
            .await
            .unwrap();
        assert_eq!(first.service_id().as_string(), SYNTHETIC);
        assert_eq!(
            first.external_addr().to_string(),
            format!("/onion3/{SYNTHETIC}:9700")
        );
        let stored = key_store::load(directory.path()).unwrap().unwrap();
        assert_eq!(stored.key_blob(), KEY_BLOB);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(directory.path().join(ONION_KEY_FILE))
                .unwrap()
                .permissions()
                .mode()
                & 0o777;
            assert_eq!(mode, 0o600);
        }
        drop(first);

        // A restart publishes the stored key, never a new one.
        let second = publish(&settings(tor.addr), directory.path(), target())
            .await
            .unwrap();
        assert_eq!(
            second.external_addr().to_string(),
            format!("/onion3/{SYNTHETIC}:9700")
        );
        assert_eq!(
            add_onion_lines(&tor),
            vec![
                "ADD_ONION NEW:ED25519-V3 Port=9700,127.0.0.1:19700\r\n".to_string(),
                format!("ADD_ONION {KEY_BLOB} Port=9700,127.0.0.1:19700\r\n"),
            ]
        );
    }

    #[tokio::test]
    async fn onion_publish_authenticates_before_adding_the_service() {
        let directory = tempfile::tempdir().unwrap();
        let password = directory.path().join("password");
        std::fs::write(&password, "pw\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&password, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let tor = FakeTor::start(FakeTorConfig::password(b"pw")).await;
        let mut with_password = settings(tor.addr);
        with_password.auth.password_file = Some(password.clone());
        let published = publish(&with_password, directory.path(), target())
            .await
            .unwrap();
        let lines = tor.received();
        let auth = lines
            .iter()
            .position(|line| line.starts_with("AUTHENTICATE "))
            .unwrap();
        let add = lines
            .iter()
            .position(|line| line.starts_with("ADD_ONION"))
            .unwrap();
        assert!(auth < add);
        assert_eq!(lines[0], "PROTOCOLINFO 1\r\n");
        drop(published);

        // No usable authentication: nothing is added, no key is created.
        let other = tempfile::tempdir().unwrap();
        let tor = FakeTor::start(FakeTorConfig::password(b"pw")).await;
        assert!(publish(&settings(tor.addr), other.path(), target())
            .await
            .is_err());
        assert!(add_onion_lines(&tor).is_empty());
        assert!(key_store::load(other.path()).unwrap().is_none());

        // Wrong password: refused by Tor, nothing added.
        std::fs::write(&password, "nope\n").unwrap();
        let tor = FakeTor::start(FakeTorConfig::password(b"pw")).await;
        let mut wrong = settings(tor.addr);
        wrong.auth.password_file = Some(password);
        assert!(publish(&wrong, other.path(), target()).await.is_err());
        assert!(add_onion_lines(&tor).is_empty());
    }

    // procfs refuses file creation, even to root: no key file exists there,
    // and none can be written.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn onion_publish_withdraws_the_service_when_the_key_cannot_be_stored() {
        let blocked = Path::new("/proc/self");
        assert!(key_store::load(blocked).unwrap().is_none());
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        assert!(publish(&settings(tor.addr), blocked, target())
            .await
            .is_err());
        assert_eq!(add_onion_lines(&tor).len(), 1);
        assert!(tor
            .received()
            .contains(&format!("DEL_ONION {SYNTHETIC}\r\n")));
    }

    #[tokio::test]
    async fn onion_publish_never_replaces_a_corrupt_key() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(ONION_KEY_FILE);
        std::fs::write(&path, b"corrupt").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        assert!(publish(&settings(tor.addr), directory.path(), target())
            .await
            .is_err());
        assert!(add_onion_lines(&tor).is_empty());
        assert_eq!(std::fs::read(&path).unwrap(), b"corrupt");
    }

    #[tokio::test]
    async fn onion_publish_targets_only_the_loopback_listener() {
        let directory = tempfile::tempdir().unwrap();
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        for bad in ["0.0.0.0:19700", "192.168.1.2:19700", "127.0.0.1:0"] {
            assert!(
                publish(&settings(tor.addr), directory.path(), bad.parse().unwrap())
                    .await
                    .is_err()
            );
        }
        assert!(add_onion_lines(&tor).is_empty());
        assert_eq!(tor.connections(), 0);
    }

    /// C-3 remainder: a guardian whose ControlPort asks for no authentication,
    /// with no method configured, never publishes (no ADD_ONION is sent) and
    /// says how to turn on cookie authentication; a plain onion node does.
    #[tokio::test]
    async fn a_guardian_never_publishes_through_a_control_port_asking_for_no_authentication() {
        let directory = tempfile::tempdir().unwrap();
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        let guardian = OnionSettings {
            mode: crate::onion::config::OnionMode::Guardian,
            ..settings(tor.addr)
        };
        match publish(&guardian, directory.path(), target()).await {
            Err(OnionError::Auth(message)) => {
                assert!(message.contains("CookieAuthentication 1"), "{message}")
            }
            Err(other) => panic!("{other}"),
            Ok(_) => panic!("a guardian published through NULL"),
        }
        assert!(add_onion_lines(&tor).is_empty());
        assert!(!tor.received().contains(&"AUTHENTICATE\r\n".to_string()));
        publish(&settings(tor.addr), directory.path(), target())
            .await
            .expect("a plain onion node, as before");
    }

    #[tokio::test]
    async fn onion_published_service_reports_closure_and_can_be_withdrawn() {
        let directory = tempfile::tempdir().unwrap();
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        let published = publish(&settings(tor.addr), directory.path(), target())
            .await
            .unwrap();
        published.withdraw().await.unwrap();
        assert!(tor
            .received()
            .contains(&format!("DEL_ONION {SYNTHETIC}\r\n")));

        let published = publish(&settings(tor.addr), directory.path(), target())
            .await
            .unwrap();
        tor.close_all();
        assert!(
            tokio::time::timeout(Duration::from_secs(5), published.closed())
                .await
                .is_ok()
        );
    }
}
