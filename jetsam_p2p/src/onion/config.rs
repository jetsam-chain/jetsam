// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Operator configuration of the onion service. Disabled by default.
//!
//! Tor must run on the same host: the ControlPort and the SOCKS port are
//! given as loopback `IP:PORT` literals (a host name would need a DNS lookup,
//! and a remote ControlPort would carry its credentials in clear), and the
//! service forwards to a loopback listener of this node.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::PathBuf;

use libp2p::{multiaddr::Protocol, Multiaddr};
use serde::{Deserialize, Serialize};

use super::OnionError;

/// Whether the onion service is offered beside the clearnet listeners or
/// instead of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnionMode {
    /// The default (D57): clearnet as before, plus the onion service. This
    /// adds reachability; it does not hide the node's IP address, which the
    /// same identity keeps announcing on clearnet.
    #[default]
    Additional,
    /// Onion only: no clearnet address is ever announced. See
    /// [`check_exclusive_mode`] for the clearnet features this excludes.
    Exclusive,
}

/// The `[network.onion]` section as written by the operator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct OnionConfig {
    /// Off unless the operator turns it on.
    pub enabled: bool,
    pub mode: OnionMode,
    /// Tor ControlPort, a loopback `IP:PORT` literal.
    pub control: String,
    /// Tor SOCKS port, a loopback `IP:PORT` literal.
    pub socks: String,
    /// Port of the node's extra loopback listener that Tor forwards to;
    /// `0` lets the system choose one at each start.
    pub local_port: u16,
    /// The port announced in `/onion3/<id>:<port>`.
    pub virtual_port: u16,
    /// Tor's control cookie (`CookieAuthentication 1`). Cookie
    /// authentication is used only with this explicit path, and only when Tor
    /// names the same file.
    pub cookie_file: Option<PathBuf>,
    /// A file holding the ControlPort password (`HashedControlPassword`),
    /// readable by its owner only.
    pub control_password_file: Option<PathBuf>,
}

pub const DEFAULT_CONTROL: &str = "127.0.0.1:9051";
pub const DEFAULT_SOCKS: &str = "127.0.0.1:9050";
pub const DEFAULT_VIRTUAL_PORT: u16 = 9700;

impl Default for OnionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: OnionMode::Additional,
            control: DEFAULT_CONTROL.to_string(),
            socks: DEFAULT_SOCKS.to_string(),
            local_port: 0,
            virtual_port: DEFAULT_VIRTUAL_PORT,
            cookie_file: None,
            control_password_file: None,
        }
    }
}

/// How the node authenticates to the ControlPort.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ControlAuthConfig {
    pub cookie_file: Option<PathBuf>,
    pub password_file: Option<PathBuf>,
}

/// A validated configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OnionSettings {
    pub mode: OnionMode,
    pub control: SocketAddr,
    pub socks: SocketAddr,
    /// `127.0.0.1:<local_port>`.
    pub local_listen: SocketAddr,
    pub virtual_port: u16,
    pub auth: ControlAuthConfig,
}

impl OnionSettings {
    /// The multiaddr of the extra loopback listener.
    pub fn local_listen_multiaddr(&self) -> Multiaddr {
        Multiaddr::empty()
            .with(match self.local_listen.ip() {
                IpAddr::V4(ip) => Protocol::Ip4(ip),
                IpAddr::V6(ip) => Protocol::Ip6(ip),
            })
            .with(Protocol::Tcp(self.local_listen.port()))
    }
}

fn loopback_endpoint(name: &str, value: &str) -> Result<SocketAddr, OnionError> {
    let addr: SocketAddr = value.parse().map_err(|_| {
        OnionError::Config(format!(
            "{name} must be an IP:PORT literal (no host name), got {value:?}"
        ))
    })?;
    if !addr.ip().is_loopback() {
        return Err(OnionError::Config(format!(
            "{name} {addr} is not a loopback address: Tor must run on this host"
        )));
    }
    if addr.port() == 0 {
        return Err(OnionError::Config(format!("{name} {addr} has port 0")));
    }
    Ok(addr)
}

fn absolute_path(name: &str, path: &Option<PathBuf>) -> Result<(), OnionError> {
    match path {
        Some(path) if !path.is_absolute() => Err(OnionError::Config(format!(
            "{name} must be an absolute path, got {}",
            path.display()
        ))),
        _ => Ok(()),
    }
}

impl OnionConfig {
    /// `Ok(None)` when disabled; the validated settings otherwise.
    pub fn validate(&self) -> Result<Option<OnionSettings>, OnionError> {
        if !self.enabled {
            return Ok(None);
        }
        let control = loopback_endpoint("onion control", &self.control)?;
        let socks = loopback_endpoint("onion socks", &self.socks)?;
        if control == socks {
            return Err(OnionError::Config(format!(
                "onion control and socks are the same endpoint {control}"
            )));
        }
        let local_listen = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), self.local_port);
        if local_listen == control || local_listen == socks {
            return Err(OnionError::Config(format!(
                "onion local_port {} is already Tor's",
                self.local_port
            )));
        }
        if self.virtual_port == 0 {
            return Err(OnionError::Config("onion virtual_port 0".into()));
        }
        absolute_path("onion cookie_file", &self.cookie_file)?;
        absolute_path("onion control_password_file", &self.control_password_file)?;
        Ok(Some(OnionSettings {
            mode: self.mode,
            control,
            socks,
            local_listen,
            virtual_port: self.virtual_port,
            auth: ControlAuthConfig {
                cookie_file: self.cookie_file.clone(),
                password_file: self.control_password_file.clone(),
            },
        }))
    }
}

fn is_loopback_only(addr: &Multiaddr) -> bool {
    match addr.iter().next() {
        Some(Protocol::Ip4(ip)) => ip.is_loopback(),
        Some(Protocol::Ip6(ip)) => ip.is_loopback(),
        _ => false,
    }
}

/// Onion-only mode refuses every clearnet feature that would expose or
/// announce this host: a listener on a non-loopback address, a declared
/// public address, a UPnP mapping and LAN discovery.
pub fn check_exclusive_mode(
    listen_addrs: &[Multiaddr],
    public_addresses: &[Multiaddr],
    upnp: bool,
    lan_discovery: bool,
) -> Result<(), OnionError> {
    if let Some(addr) = listen_addrs.iter().find(|addr| !is_loopback_only(addr)) {
        return Err(OnionError::Config(format!(
            "onion-only mode: P2P listener {addr} is not loopback"
        )));
    }
    if let Some(addr) = public_addresses.first() {
        return Err(OnionError::Config(format!(
            "onion-only mode: public address {addr} would be announced"
        )));
    }
    if upnp {
        return Err(OnionError::Config(
            "onion-only mode: UPnP would map a clearnet port".into(),
        ));
    }
    if lan_discovery {
        return Err(OnionError::Config(
            "onion-only mode: LAN discovery would broadcast this node".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enabled() -> OnionConfig {
        OnionConfig {
            enabled: true,
            ..OnionConfig::default()
        }
    }

    #[test]
    fn onion_config_is_disabled_by_default_and_validates_to_nothing() {
        let config = OnionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.mode, OnionMode::Additional);
        assert_eq!(config.validate().unwrap(), None);
        // A disabled section is never an obstacle, whatever it holds.
        let config = OnionConfig {
            control: "not an address".into(),
            ..OnionConfig::default()
        };
        assert_eq!(config.validate().unwrap(), None);
    }

    #[test]
    fn onion_config_parses_from_toml_like_json_with_defaults() {
        let config: OnionConfig =
            serde_json::from_str(r#"{"enabled": true, "mode": "exclusive"}"#).unwrap();
        assert!(config.enabled);
        assert_eq!(config.mode, OnionMode::Exclusive);
        assert_eq!(config.control, DEFAULT_CONTROL);
        assert_eq!(config.socks, DEFAULT_SOCKS);
        assert_eq!(config.virtual_port, DEFAULT_VIRTUAL_PORT);
        let config: OnionConfig = serde_json::from_str("{}").unwrap();
        assert_eq!(config, OnionConfig::default());
        assert!(serde_json::from_str::<OnionConfig>(r#"{"mode": "onion-only"}"#).is_err());
    }

    #[test]
    fn onion_config_validates_the_default_loopback_endpoints() {
        let settings = OnionConfig {
            local_port: 19700,
            cookie_file: Some("/run/tor/control.authcookie".into()),
            ..enabled()
        }
        .validate()
        .unwrap()
        .unwrap();
        assert_eq!(settings.control, "127.0.0.1:9051".parse().unwrap());
        assert_eq!(settings.socks, "127.0.0.1:9050".parse().unwrap());
        assert_eq!(settings.local_listen, "127.0.0.1:19700".parse().unwrap());
        assert_eq!(settings.virtual_port, 9700);
        assert_eq!(
            settings.local_listen_multiaddr().to_string(),
            "/ip4/127.0.0.1/tcp/19700"
        );
        assert_eq!(
            settings.auth.cookie_file,
            Some(PathBuf::from("/run/tor/control.authcookie"))
        );
        let ipv6 = OnionConfig {
            control: "[::1]:9051".into(),
            socks: "[::1]:9050".into(),
            ..enabled()
        }
        .validate()
        .unwrap()
        .unwrap();
        assert!(ipv6.control.ip().is_loopback());
    }

    #[test]
    fn onion_config_refuses_remote_named_or_ambiguous_endpoints() {
        let refused = [
            // Remote Tor: credentials and traffic would cross the network.
            OnionConfig {
                control: "192.168.1.2:9051".into(),
                ..enabled()
            },
            OnionConfig {
                socks: "10.0.0.1:9050".into(),
                ..enabled()
            },
            OnionConfig {
                socks: "0.0.0.0:9050".into(),
                ..enabled()
            },
            // A host name would be resolved through DNS.
            OnionConfig {
                control: "localhost:9051".into(),
                ..enabled()
            },
            OnionConfig {
                socks: "localhost:9050".into(),
                ..enabled()
            },
            OnionConfig {
                socks: "127.0.0.1:0".into(),
                ..enabled()
            },
            OnionConfig {
                socks: DEFAULT_CONTROL.into(),
                ..enabled()
            },
            OnionConfig {
                virtual_port: 0,
                ..enabled()
            },
            OnionConfig {
                local_port: 9050,
                ..enabled()
            },
            OnionConfig {
                local_port: 9051,
                ..enabled()
            },
            OnionConfig {
                cookie_file: Some("relative/cookie".into()),
                ..enabled()
            },
            OnionConfig {
                control_password_file: Some("password".into()),
                ..enabled()
            },
        ];
        for config in refused {
            assert!(config.validate().is_err(), "{config:?} accepted");
        }
    }

    #[test]
    fn exclusive_mode_refuses_every_clearnet_exposure() {
        let loopback: Vec<Multiaddr> = vec![
            "/ip4/127.0.0.1/tcp/9700".parse().unwrap(),
            "/ip6/::1/tcp/9700".parse().unwrap(),
        ];
        assert!(check_exclusive_mode(&loopback, &[], false, false).is_ok());
        assert!(check_exclusive_mode(&[], &[], false, false).is_ok());

        let wildcard: Vec<Multiaddr> = vec!["/ip4/0.0.0.0/tcp/9700".parse().unwrap()];
        assert!(check_exclusive_mode(&wildcard, &[], false, false).is_err());
        let lan: Vec<Multiaddr> = vec!["/ip4/192.168.1.5/tcp/9700".parse().unwrap()];
        assert!(check_exclusive_mode(&lan, &[], false, false).is_err());
        let named: Vec<Multiaddr> = vec!["/dns4/localhost/tcp/9700".parse().unwrap()];
        assert!(check_exclusive_mode(&named, &[], false, false).is_err());
        let public: Vec<Multiaddr> = vec!["/ip4/8.8.8.8/tcp/9700".parse().unwrap()];
        assert!(check_exclusive_mode(&loopback, &public, false, false).is_err());
        assert!(check_exclusive_mode(&loopback, &[], true, false).is_err());
        assert!(check_exclusive_mode(&loopback, &[], false, true).is_err());
    }
}
