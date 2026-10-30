// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A bounded client for Tor's ControlPort (control-spec, version 1).
//!
//! Only four commands are used: `PROTOCOLINFO 1`, `AUTHENTICATE`,
//! `ADD_ONION` and `DEL_ONION`. Every reply line is bounded in length, every
//! reply in line count, every round trip in time, and nothing read from the
//! port is trusted before it is parsed: a service ID must carry a valid v3
//! checksum and a key must have the exact `ED25519-V3:<base64>` shape before
//! either is stored or used.
//!
//! Secrets are sent hex-encoded (`AUTHENTICATE <hex>`), so no quoting is
//! ever needed on the way out; any command line holding CR, LF or NUL is
//! refused before it is written. Quoted strings coming back from Tor (the
//! cookie file path) are decoded with the C escapes Tor uses.
//!
//! Cookie authentication reads only the file the operator configured, and
//! only when Tor announces that same file: a process impersonating Tor on the
//! ControlPort can therefore not make the node read and send an arbitrary
//! 32-byte file. (`SAFECOOKIE` would avoid sending the cookie at all but needs
//! HMAC-SHA256, which this crate does not depend on.)

use std::io::Read;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use super::address::OnionServiceId;
use super::config::ControlAuthConfig;
use super::key_store::OnionKey;
use super::OnionError;

/// Bound on one control round trip.
pub const CONTROL_TIMEOUT: Duration = Duration::from_secs(10);
/// Bound on one reply line (a cookie path can be as long as `PATH_MAX`).
pub const MAX_LINE_BYTES: usize = 8192;
/// Bound on the lines of one reply, data lines included.
pub const MAX_REPLY_LINES: usize = 64;
const COOKIE_LEN: usize = 32;
const MAX_PASSWORD_BYTES: usize = 1024;
const MAX_MESSAGE_CHARS: usize = 200;

/// One line of a reply: its text after `NNN-`/`NNN+`/`NNN `, and for a
/// `NNN+` line the data block that follows it (dot-unescaped).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplyLine {
    pub text: String,
    pub data: Vec<String>,
}

/// A complete reply: one status code shared by all its lines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reply {
    pub code: u16,
    pub lines: Vec<ReplyLine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthMethod {
    Null,
    HashedPassword,
    Cookie,
    SafeCookie,
    Other(String),
}

/// What `PROTOCOLINFO 1` says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProtocolInfo {
    pub methods: Vec<AuthMethod>,
    pub cookie_file: Option<PathBuf>,
    pub tor_version: Option<String>,
}

/// The authentication the node will perform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthChoice {
    Null,
    Cookie(PathBuf),
    Password(PathBuf),
}

/// Pick an authentication from what Tor offers and what the operator set.
///
/// `NULL` when Tor asks for nothing; otherwise the configured password when
/// Tor accepts passwords; otherwise the configured cookie file, provided Tor
/// names that very file. Tor never chooses which file the node reads.
pub fn choose_auth(
    info: &ProtocolInfo,
    auth: &ControlAuthConfig,
) -> Result<AuthChoice, OnionError> {
    let offers = |method: &AuthMethod| info.methods.contains(method);
    if offers(&AuthMethod::Null) {
        return Ok(AuthChoice::Null);
    }
    if let Some(password) = &auth.password_file {
        if offers(&AuthMethod::HashedPassword) {
            return Ok(AuthChoice::Password(password.clone()));
        }
    }
    if let Some(cookie) = &auth.cookie_file {
        if offers(&AuthMethod::Cookie) {
            return match &info.cookie_file {
                Some(announced) if announced == cookie => Ok(AuthChoice::Cookie(cookie.clone())),
                Some(announced) => Err(OnionError::Auth(format!(
                    "Tor names the cookie file {}, the configuration {}; refusing to read a file Tor chose",
                    printable(&announced.display().to_string()),
                    cookie.display()
                ))),
                None => Err(OnionError::Auth(
                    "Tor offers cookie authentication without naming its cookie file".into(),
                )),
            };
        }
    }
    Err(OnionError::Auth(format!(
        "no usable authentication: Tor offers {:?}; configure control_password_file (HASHEDPASSWORD) or cookie_file (COOKIE)",
        info.methods
    )))
}

fn protocol(message: impl Into<String>) -> OnionError {
    OnionError::ControlProtocol(message.into())
}

/// Text from Tor as it may appear in a log: bounded, without control
/// characters.
fn printable(text: &str) -> String {
    text.chars()
        .take(MAX_MESSAGE_CHARS)
        .map(|c| if c.is_control() { '?' } else { c })
        .collect()
}

/// `AUTHENTICATE <hex>`: hex needs no quoting, whatever the secret holds.
fn authenticate_line(secret: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let mut line = String::with_capacity(13 + secret.len() * 2);
    line.push_str("AUTHENTICATE ");
    for byte in secret {
        line.push(char::from(DIGITS[usize::from(byte >> 4)]));
        line.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    line
}

fn zeroize_string(text: String) {
    let mut bytes = text.into_bytes();
    bytes.fill(0);
}

/// Split `NNN<sep>text` into its code, separator and text.
pub(crate) fn parse_reply_line(line: &str) -> Result<(u16, char, &str), OnionError> {
    let bytes = line.as_bytes();
    if bytes.len() < 4 || !bytes[..3].iter().all(u8::is_ascii_digit) {
        return Err(protocol(format!(
            "malformed reply line {:?}",
            printable(line)
        )));
    }
    let code = bytes[..3]
        .iter()
        .fold(0u16, |code, digit| code * 10 + u16::from(digit - b'0'));
    if !(200..700).contains(&code) {
        return Err(protocol(format!("unknown status code {code}")));
    }
    let separator = match bytes[3] {
        b'-' => '-',
        b'+' => '+',
        b' ' => ' ',
        _ => {
            return Err(protocol(format!(
                "malformed reply separator in {:?}",
                printable(line)
            )))
        }
    };
    Ok((code, separator, &line[4..]))
}

/// Decode a quoted string starting at `text[0] == '"'`; returns the decoded
/// bytes and the text after the closing quote.
pub(crate) fn unquote(text: &str) -> Result<(Vec<u8>, &str), OnionError> {
    let bytes = text.as_bytes();
    if bytes.first() != Some(&b'"') {
        return Err(protocol("expected a quoted string"));
    }
    let mut decoded = Vec::new();
    let mut index = 1;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            b'"' => return Ok((decoded, &text[index + 1..])),
            b'\\' => {
                let escaped = *bytes
                    .get(index + 1)
                    .ok_or_else(|| protocol("unterminated escape in a quoted string"))?;
                index += 2;
                match escaped {
                    b'n' => decoded.push(b'\n'),
                    b't' => decoded.push(b'\t'),
                    b'r' => decoded.push(b'\r'),
                    b'\\' | b'"' | b'\'' => decoded.push(escaped),
                    b'0'..=b'7' => {
                        let mut value = u32::from(escaped - b'0');
                        for _ in 0..2 {
                            match bytes.get(index) {
                                Some(digit @ b'0'..=b'7') => {
                                    value = value * 8 + u32::from(digit - b'0');
                                    index += 1;
                                }
                                _ => break,
                            }
                        }
                        decoded.push(
                            u8::try_from(value)
                                .map_err(|_| protocol("octal escape out of range"))?,
                        );
                    }
                    _ => return Err(protocol("unknown escape in a quoted string")),
                }
            }
            _ => {
                decoded.push(byte);
                index += 1;
            }
        }
    }
    Err(protocol("unterminated quoted string"))
}

/// `KEY=value KEY="quoted value" FLAG` arguments of a reply line.
fn keyword_args(mut text: &str) -> Result<Vec<(&str, Vec<u8>)>, OnionError> {
    let mut args = Vec::new();
    loop {
        text = text.trim_start_matches(' ');
        if text.is_empty() {
            return Ok(args);
        }
        let key_end = text.find(['=', ' ']).unwrap_or(text.len());
        let key = &text[..key_end];
        text = &text[key_end..];
        let Some(value) = text.strip_prefix('=') else {
            args.push((key, Vec::new()));
            continue;
        };
        if value.starts_with('"') {
            let (decoded, rest) = unquote(value)?;
            args.push((key, decoded));
            text = rest;
        } else {
            let end = value.find(' ').unwrap_or(value.len());
            args.push((key, value.as_bytes()[..end].to_vec()));
            text = &value[end..];
        }
    }
}

fn path_from_bytes(bytes: Vec<u8>) -> PathBuf {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        PathBuf::from(std::ffi::OsString::from_vec(bytes))
    }
    #[cfg(not(unix))]
    {
        PathBuf::from(String::from_utf8_lossy(&bytes).into_owned())
    }
}

impl AuthMethod {
    fn from_name(name: &str) -> Self {
        match name {
            "NULL" => Self::Null,
            "HASHEDPASSWORD" => Self::HashedPassword,
            "COOKIE" => Self::Cookie,
            "SAFECOOKIE" => Self::SafeCookie,
            other => Self::Other(printable(other)),
        }
    }
}

/// Parse the reply to `PROTOCOLINFO 1`.
pub fn parse_protocol_info(reply: &Reply) -> Result<ProtocolInfo, OnionError> {
    let mut methods = None;
    let mut cookie_file = None;
    let mut tor_version = None;
    for line in &reply.lines {
        let (keyword, rest) = line.text.split_once(' ').unwrap_or((&line.text, ""));
        match keyword {
            "AUTH" => {
                for (key, value) in keyword_args(rest)? {
                    match key {
                        "METHODS" => {
                            methods = Some(
                                String::from_utf8_lossy(&value)
                                    .split(',')
                                    .filter(|name| !name.is_empty())
                                    .map(AuthMethod::from_name)
                                    .collect(),
                            )
                        }
                        "COOKIEFILE" => cookie_file = Some(path_from_bytes(value)),
                        _ => {}
                    }
                }
            }
            "VERSION" => {
                for (key, value) in keyword_args(rest)? {
                    if key == "Tor" {
                        tor_version = Some(printable(&String::from_utf8_lossy(&value)));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(ProtocolInfo {
        methods: methods.ok_or_else(|| protocol("PROTOCOLINFO reply without AUTH METHODS"))?,
        cookie_file,
        tor_version,
    })
}

fn open_secret_file(path: &Path) -> Result<std::fs::File, OnionError> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let file = options
        .open(path)
        .map_err(|error| OnionError::Auth(format!("open {}: {error}", path.display())))?;
    let metadata = file
        .metadata()
        .map_err(|error| OnionError::Auth(format!("stat {}: {error}", path.display())))?;
    if !metadata.is_file() {
        return Err(OnionError::Auth(format!(
            "{} is not a regular file",
            path.display()
        )));
    }
    Ok(file)
}

fn read_bounded(path: &Path, file: std::fs::File, limit: usize) -> Result<Vec<u8>, OnionError> {
    let mut bytes = Vec::with_capacity(limit + 1);
    if let Err(error) = file.take(limit as u64 + 1).read_to_end(&mut bytes) {
        bytes.fill(0);
        return Err(OnionError::Auth(format!(
            "read {}: {error}",
            path.display()
        )));
    }
    Ok(bytes)
}

/// Tor's cookie: exactly 32 bytes.
fn read_cookie(path: &Path) -> Result<Vec<u8>, OnionError> {
    let file = open_secret_file(path)?;
    let mut cookie = read_bounded(path, file, COOKIE_LEN)?;
    if cookie.len() != COOKIE_LEN {
        cookie.fill(0);
        return Err(OnionError::Auth(format!(
            "{} is not a {COOKIE_LEN}-byte Tor control cookie",
            path.display()
        )));
    }
    Ok(cookie)
}

/// The password file: owner-only, bounded, one trailing newline removed.
fn read_password(path: &Path) -> Result<Vec<u8>, OnionError> {
    let file = open_secret_file(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let metadata = file
            .metadata()
            .map_err(|error| OnionError::Auth(format!("stat {}: {error}", path.display())))?;
        let mode = metadata.permissions().mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(OnionError::Auth(format!(
                "{} is readable by others (mode {mode:o}); expected owner-only",
                path.display()
            )));
        }
        // SAFETY: geteuid has no preconditions and does not access pointers.
        let expected_owner = unsafe { libc::geteuid() };
        if metadata.uid() != expected_owner {
            return Err(OnionError::Auth(format!(
                "{} belongs to uid {}, expected {expected_owner}",
                path.display(),
                metadata.uid()
            )));
        }
    }
    let mut password = read_bounded(path, file, MAX_PASSWORD_BYTES)?;
    if password.len() > MAX_PASSWORD_BYTES {
        password.fill(0);
        return Err(OnionError::Auth(format!(
            "{} exceeds {MAX_PASSWORD_BYTES} bytes",
            path.display()
        )));
    }
    if password.last() == Some(&b'\n') {
        password.pop();
        if password.last() == Some(&b'\r') {
            password.pop();
        }
    }
    if password.is_empty() {
        return Err(OnionError::Auth(format!("{} is empty", path.display())));
    }
    Ok(password)
}

/// Which key `ADD_ONION` publishes.
pub enum KeyRequest<'a> {
    /// Let Tor generate a key and return it once.
    New,
    /// Publish the stored key again.
    Existing(&'a OnionKey),
}

/// The published service; `new_key` is set for [`KeyRequest::New`].
#[derive(Debug)]
pub struct AddOnionReply {
    pub service_id: OnionServiceId,
    pub new_key: Option<OnionKey>,
}

/// One authenticated (or about to be) ControlPort session.
pub struct TorControl {
    stream: BufReader<TcpStream>,
    timeout: Duration,
}

impl TorControl {
    /// Connect to a loopback ControlPort.
    pub async fn connect(addr: SocketAddr) -> Result<Self, OnionError> {
        if !addr.ip().is_loopback() {
            return Err(OnionError::Config(format!(
                "refusing the non-loopback ControlPort {addr}"
            )));
        }
        let stream = tokio::time::timeout(CONTROL_TIMEOUT, TcpStream::connect(addr))
            .await
            .map_err(|_| OnionError::Timeout("connecting to the Tor ControlPort"))??;
        Ok(Self {
            stream: BufReader::new(stream),
            timeout: CONTROL_TIMEOUT,
        })
    }

    /// Change the bound on each round trip.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Send one command line and read its reply; only `250` is success.
    pub async fn command(&mut self, line: &str) -> Result<Reply, OnionError> {
        if line.bytes().any(|byte| matches!(byte, b'\r' | b'\n' | 0)) {
            return Err(protocol("refusing a command line holding CR, LF or NUL"));
        }
        let mut wire = Vec::with_capacity(line.len() + 2);
        wire.extend_from_slice(line.as_bytes());
        wire.extend_from_slice(b"\r\n");
        let timeout = self.timeout;
        let round_trip = async {
            let written = async {
                let stream = self.stream.get_mut();
                stream.write_all(&wire).await?;
                stream.flush().await
            }
            .await;
            wire.fill(0);
            written?;
            self.read_reply().await
        };
        let reply = tokio::time::timeout(timeout, round_trip)
            .await
            .map_err(|_| OnionError::Timeout("a Tor control command"))??;
        if reply.code != 250 {
            let message = reply
                .lines
                .last()
                .map(|line| printable(&line.text))
                .unwrap_or_default();
            return Err(OnionError::ControlRefused {
                code: reply.code,
                message,
            });
        }
        Ok(reply)
    }

    /// One line, at most [`MAX_LINE_BYTES`], without its CRLF (a bare LF is
    /// tolerated).
    async fn read_line(&mut self) -> Result<String, OnionError> {
        let mut buffer = Vec::new();
        let read = (&mut self.stream)
            .take(MAX_LINE_BYTES as u64 + 1)
            .read_until(b'\n', &mut buffer)
            .await?;
        if read == 0 {
            return Err(OnionError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Tor closed the control connection",
            )));
        }
        if buffer.last() != Some(&b'\n') {
            if buffer.len() > MAX_LINE_BYTES {
                return Err(protocol("reply line exceeds the length bound"));
            }
            return Err(OnionError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Tor closed the control connection mid-line",
            )));
        }
        buffer.pop();
        if buffer.last() == Some(&b'\r') {
            buffer.pop();
        }
        String::from_utf8(buffer).map_err(|_| protocol("reply line is not UTF-8"))
    }

    async fn read_reply(&mut self) -> Result<Reply, OnionError> {
        let mut code = None;
        let mut lines = Vec::new();
        let mut count = 0usize;
        loop {
            let line = self.read_line().await?;
            count += 1;
            if count > MAX_REPLY_LINES {
                return Err(protocol("reply exceeds the line bound"));
            }
            let (line_code, separator, text) = parse_reply_line(&line)?;
            if let Some(previous) = code {
                if previous != line_code {
                    return Err(protocol(format!(
                        "reply mixes status codes {previous} and {line_code}"
                    )));
                }
            }
            code = Some(line_code);
            let mut entry = ReplyLine {
                text: text.to_string(),
                data: Vec::new(),
            };
            match separator {
                '-' => lines.push(entry),
                '+' => {
                    loop {
                        let data = self.read_line().await?;
                        count += 1;
                        if count > MAX_REPLY_LINES {
                            return Err(protocol("reply exceeds the line bound"));
                        }
                        if data == "." {
                            break;
                        }
                        let unescaped = if data.starts_with('.') {
                            data[1..].to_string()
                        } else {
                            data
                        };
                        entry.data.push(unescaped);
                    }
                    lines.push(entry);
                }
                _ => {
                    lines.push(entry);
                    return Ok(Reply {
                        code: line_code,
                        lines,
                    });
                }
            }
        }
    }

    pub async fn protocol_info(&mut self) -> Result<ProtocolInfo, OnionError> {
        let reply = self.command("PROTOCOLINFO 1").await?;
        parse_protocol_info(&reply)
    }

    pub async fn authenticate(&mut self, choice: &AuthChoice) -> Result<(), OnionError> {
        let line = match choice {
            AuthChoice::Null => "AUTHENTICATE".to_string(),
            AuthChoice::Cookie(path) => {
                let mut cookie = read_cookie(path)?;
                let line = authenticate_line(&cookie);
                cookie.fill(0);
                line
            }
            AuthChoice::Password(path) => {
                let mut password = read_password(path)?;
                let line = authenticate_line(&password);
                password.fill(0);
                line
            }
        };
        let result = self.command(&line).await;
        zeroize_string(line);
        match result {
            Ok(_) => Ok(()),
            Err(OnionError::ControlRefused { code, message }) => Err(OnionError::Auth(format!(
                "Tor refused authentication ({code}): {message}"
            ))),
            Err(other) => Err(other),
        }
    }

    /// `ADD_ONION <key> Port=<virtual_port>,<target>`; the service lives as
    /// long as this control connection.
    pub async fn add_onion(
        &mut self,
        key: KeyRequest<'_>,
        virtual_port: u16,
        target: SocketAddr,
    ) -> Result<AddOnionReply, OnionError> {
        if virtual_port == 0 {
            return Err(OnionError::Config("onion virtual port 0".into()));
        }
        if !target.ip().is_loopback() || target.port() == 0 {
            return Err(OnionError::Config(format!(
                "the onion service must forward to a loopback listener, not {target}"
            )));
        }
        let key_arg = match &key {
            KeyRequest::New => "NEW:ED25519-V3",
            KeyRequest::Existing(stored) => stored.key_blob(),
        };
        let reply = self
            .command(&format!("ADD_ONION {key_arg} Port={virtual_port},{target}"))
            .await?;

        let mut service_id = None;
        let mut private_key = None;
        for line in &reply.lines {
            let (slot, value) = if let Some(value) = line.text.strip_prefix("ServiceID=") {
                (&mut service_id, value)
            } else if let Some(value) = line.text.strip_prefix("PrivateKey=") {
                (&mut private_key, value)
            } else {
                continue;
            };
            if slot.replace(value.to_string()).is_some() {
                return Err(protocol("ADD_ONION reply repeats a field"));
            }
        }
        let service_id = service_id
            .ok_or_else(|| protocol("ADD_ONION reply without ServiceID"))
            .and_then(|text| OnionServiceId::parse(&text))?;

        match key {
            KeyRequest::New => {
                let new_key = match private_key {
                    Some(blob) => {
                        let parsed = OnionKey::new(&blob, service_id.clone());
                        zeroize_string(blob);
                        parsed
                    }
                    None => Err(protocol("ADD_ONION NEW reply without PrivateKey")),
                };
                match new_key {
                    Ok(new_key) => Ok(AddOnionReply {
                        service_id,
                        new_key: Some(new_key),
                    }),
                    Err(error) => {
                        let _ = self.del_onion(&service_id).await;
                        Err(error)
                    }
                }
            }
            KeyRequest::Existing(stored) => {
                if let Some(blob) = private_key {
                    zeroize_string(blob);
                }
                if &service_id != stored.service_id() {
                    let _ = self.del_onion(&service_id).await;
                    return Err(OnionError::KeyStore(format!(
                        "Tor derived {service_id} from the key stored for {}",
                        stored.service_id()
                    )));
                }
                Ok(AddOnionReply {
                    service_id,
                    new_key: None,
                })
            }
        }
    }

    pub async fn del_onion(&mut self, service_id: &OnionServiceId) -> Result<(), OnionError> {
        self.command(&format!("DEL_ONION {service_id}"))
            .await
            .map(|_| ())
    }

    /// Wait until Tor closes the session (which also removes the service).
    /// Output nobody asked for is drained, line by bounded line.
    pub async fn wait_closed(&mut self) -> OnionError {
        loop {
            match self.read_line().await {
                Err(OnionError::Io(error)) => return OnionError::Io(error),
                Ok(_) | Err(_) => continue,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onion::key_store::tests::{sample_key, KEY_BLOB, SYNTHETIC};
    use crate::onion::test_support::{FakeTor, FakeTorConfig, ScriptedServer};

    fn info(methods: Vec<AuthMethod>, cookie_file: Option<&str>) -> ProtocolInfo {
        ProtocolInfo {
            methods,
            cookie_file: cookie_file.map(PathBuf::from),
            tor_version: None,
        }
    }

    fn reply(code: u16, texts: &[&str]) -> Reply {
        Reply {
            code,
            lines: texts
                .iter()
                .map(|text| ReplyLine {
                    text: text.to_string(),
                    data: Vec::new(),
                })
                .collect(),
        }
    }

    #[test]
    fn control_reply_lines_are_split_strictly() {
        assert_eq!(parse_reply_line("250 OK").unwrap(), (250, ' ', "OK"));
        assert_eq!(
            parse_reply_line("250-ServiceID=x").unwrap(),
            (250, '-', "ServiceID=x")
        );
        assert_eq!(parse_reply_line("250+data=").unwrap(), (250, '+', "data="));
        assert_eq!(parse_reply_line("515 Bad").unwrap(), (515, ' ', "Bad"));
        for bad in [
            "", "250", "25 OK", "2a0 OK", "250*OK", "100 OK", "750 OK", "-250 x",
        ] {
            assert!(parse_reply_line(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn control_quoted_strings_are_unescaped_like_tor_writes_them() {
        let (value, rest) = unquote(r#""/a b/\"q\"/\\x\n\t\r\'\101" tail"#).unwrap();
        assert_eq!(value, b"/a b/\"q\"/\\x\n\t\r'A".to_vec());
        assert_eq!(rest, " tail");
        let (value, rest) = unquote(r#""""#).unwrap();
        assert!(value.is_empty());
        assert_eq!(rest, "");
        for bad in [
            r#""unterminated"#,
            r#""bad \q escape""#,
            r#"no quote"#,
            r#""\"#,
            r#""\400""#,
        ] {
            assert!(unquote(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn control_protocolinfo_is_parsed_with_quoted_cookie_paths() {
        let parsed = parse_protocol_info(&reply(
            250,
            &[
                "PROTOCOLINFO 1",
                r#"AUTH METHODS=COOKIE,SAFECOOKIE,HASHEDPASSWORD COOKIEFILE="/run/tor dir/\"c\"\\control.authcookie""#,
                r#"VERSION Tor="0.4.8.12""#,
                "OK",
            ],
        ))
        .unwrap();
        assert_eq!(
            parsed.methods,
            vec![
                AuthMethod::Cookie,
                AuthMethod::SafeCookie,
                AuthMethod::HashedPassword
            ]
        );
        assert_eq!(
            parsed.cookie_file,
            Some(PathBuf::from(r#"/run/tor dir/"c"\control.authcookie"#))
        );
        assert_eq!(parsed.tor_version.as_deref(), Some("0.4.8.12"));

        let null = parse_protocol_info(&reply(250, &["PROTOCOLINFO 1", "AUTH METHODS=NULL", "OK"]))
            .unwrap();
        assert_eq!(null.methods, vec![AuthMethod::Null]);
        assert_eq!(null.cookie_file, None);

        assert!(parse_protocol_info(&reply(250, &["PROTOCOLINFO 1", "OK"])).is_err());
        assert!(parse_protocol_info(&reply(
            250,
            &[
                "PROTOCOLINFO 1",
                r#"AUTH METHODS=COOKIE COOKIEFILE="/unterminated"#,
                "OK"
            ]
        ))
        .is_err());
    }

    #[test]
    fn control_auth_choice_follows_protocolinfo_and_the_operator() {
        let cookie = ControlAuthConfig {
            cookie_file: Some("/run/tor/control.authcookie".into()),
            password_file: None,
        };
        let password = ControlAuthConfig {
            cookie_file: None,
            password_file: Some("/etc/jetsam/tor-password".into()),
        };
        let both = ControlAuthConfig {
            cookie_file: cookie.cookie_file.clone(),
            password_file: password.password_file.clone(),
        };
        let none = ControlAuthConfig::default();
        let tor_cookie = Some("/run/tor/control.authcookie");

        assert_eq!(
            choose_auth(&info(vec![AuthMethod::Null], None), &none).unwrap(),
            AuthChoice::Null
        );
        assert_eq!(
            choose_auth(
                &info(vec![AuthMethod::Cookie, AuthMethod::SafeCookie], tor_cookie),
                &cookie
            )
            .unwrap(),
            AuthChoice::Cookie("/run/tor/control.authcookie".into())
        );
        assert_eq!(
            choose_auth(&info(vec![AuthMethod::HashedPassword], None), &password).unwrap(),
            AuthChoice::Password("/etc/jetsam/tor-password".into())
        );
        assert_eq!(
            choose_auth(
                &info(
                    vec![AuthMethod::Cookie, AuthMethod::HashedPassword],
                    tor_cookie
                ),
                &both
            )
            .unwrap(),
            AuthChoice::Password("/etc/jetsam/tor-password".into())
        );

        // Tor names another cookie file: never read a file Tor chose.
        assert!(choose_auth(
            &info(
                vec![AuthMethod::Cookie],
                Some("/home/node/.jetsam/wallet.key")
            ),
            &cookie
        )
        .is_err());
        assert!(choose_auth(&info(vec![AuthMethod::Cookie], None), &cookie).is_err());
        // Cookie offered but no path configured.
        assert!(choose_auth(&info(vec![AuthMethod::Cookie], tor_cookie), &none).is_err());
        assert!(choose_auth(&info(vec![AuthMethod::Cookie], tor_cookie), &password).is_err());
        assert!(choose_auth(&info(vec![AuthMethod::HashedPassword], None), &cookie).is_err());
        assert!(choose_auth(&info(vec![AuthMethod::SafeCookie], tor_cookie), &cookie).is_err());
        assert!(choose_auth(&info(vec![], None), &both).is_err());
    }

    #[tokio::test]
    async fn control_multi_line_and_data_replies_are_read_completely() {
        let server = ScriptedServer::start(vec![
            "250-first\r\n250+block=\r\nline one\r\n..dotted\r\n.\r\n250 OK\r\n".into(),
        ])
        .await;
        let mut control = TorControl::connect(server.addr).await.unwrap();
        let reply = control.command("GETINFO block").await.unwrap();
        assert_eq!(reply.code, 250);
        assert_eq!(reply.lines.len(), 3);
        assert_eq!(reply.lines[0].text, "first");
        assert_eq!(reply.lines[1].text, "block=");
        assert_eq!(
            reply.lines[1].data,
            vec!["line one".to_string(), ".dotted".to_string()]
        );
        assert_eq!(reply.lines[2].text, "OK");
        assert_eq!(server.received(), vec!["GETINFO block\r\n".to_string()]);
    }

    #[tokio::test]
    async fn control_error_replies_are_errors() {
        for (raw, code) in [
            ("552 Unrecognized key\r\n", 552u16),
            ("250-partial\r\n510 Unrecognized command\r\n", 0),
            ("451 Resource exhausted\r\n", 451),
            ("650 STATUS_GENERAL NOTICE\r\n", 650),
        ] {
            let server = ScriptedServer::start(vec![raw.into()]).await;
            let mut control = TorControl::connect(server.addr).await.unwrap();
            let error = control.command("GETINFO x").await.unwrap_err();
            match error {
                OnionError::ControlRefused { code: got, .. } => assert_eq!(got, code, "{raw:?}"),
                OnionError::ControlProtocol(_) => assert_eq!(code, 0, "{raw:?}"),
                other => panic!("{raw:?}: unexpected {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn control_replies_are_bounded_in_length_lines_time_and_completeness() {
        let long = format!("250 {}\r\n", "x".repeat(MAX_LINE_BYTES));
        let many: String = (0..=MAX_REPLY_LINES)
            .map(|_| "250-x\r\n")
            .collect::<String>()
            + "250 OK\r\n";
        let endless_data =
            "250+d=\r\n".to_string() + &(0..=MAX_REPLY_LINES).map(|_| "x\r\n").collect::<String>();
        for raw in [
            long,
            many,
            endless_data,
            "250-cut".to_string(),
            "250-a\r\n".to_string(),
        ] {
            let server = ScriptedServer::start(vec![raw.clone()]).await;
            let mut control = TorControl::connect(server.addr).await.unwrap();
            assert!(control.command("GETINFO x").await.is_err(), "{raw:.40?}");
        }
        let silent = ScriptedServer::start(vec![]).await;
        let mut control = TorControl::connect(silent.addr)
            .await
            .unwrap()
            .with_timeout(Duration::from_millis(200));
        assert!(matches!(
            control.command("GETINFO x").await,
            Err(OnionError::Timeout(_))
        ));
    }

    #[tokio::test]
    async fn control_command_lines_cannot_carry_line_breaks() {
        let server = ScriptedServer::start(vec!["250 OK\r\n".into()]).await;
        let mut control = TorControl::connect(server.addr).await.unwrap();
        for bad in ["GETINFO x\r\nDEL_ONION y", "GETINFO x\n", "GETINFO \0"] {
            assert!(control.command(bad).await.is_err(), "{bad:?} sent");
        }
        assert!(server.received().is_empty());
    }

    #[tokio::test]
    async fn control_refuses_a_non_loopback_port_without_connecting() {
        assert!(matches!(
            TorControl::connect("192.0.2.1:9051".parse().unwrap()).await,
            Err(OnionError::Config(_))
        ));
    }

    #[tokio::test]
    async fn control_authenticates_with_null_cookie_and_password() {
        // NULL.
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        let info = control.protocol_info().await.unwrap();
        assert_eq!(
            choose_auth(&info, &ControlAuthConfig::default()).unwrap(),
            AuthChoice::Null
        );
        control.authenticate(&AuthChoice::Null).await.unwrap();
        assert!(tor.received().contains(&"AUTHENTICATE\r\n".to_string()));

        // Cookie: the configured file, hex-encoded.
        let directory = tempfile::tempdir().unwrap();
        let cookie_path = directory.path().join("control \"auth\" cookie");
        let cookie: [u8; 32] = std::array::from_fn(|index| index as u8 ^ 0x5a);
        std::fs::write(&cookie_path, cookie).unwrap();
        let tor = FakeTor::start(FakeTorConfig::cookie(&cookie_path, cookie)).await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        let info = control.protocol_info().await.unwrap();
        assert_eq!(info.cookie_file.as_deref(), Some(cookie_path.as_path()));
        let auth = ControlAuthConfig {
            cookie_file: Some(cookie_path.clone()),
            password_file: None,
        };
        let choice = choose_auth(&info, &auth).unwrap();
        control.authenticate(&choice).await.unwrap();
        let hex: String = cookie.iter().map(|byte| format!("{byte:02X}")).collect();
        assert!(tor
            .received()
            .iter()
            .any(|line| line.eq_ignore_ascii_case(&format!("AUTHENTICATE {hex}\r\n"))));

        // A cookie file of the wrong size is never sent.
        std::fs::write(&cookie_path, [1u8; 31]).unwrap();
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        assert!(control.authenticate(&choice).await.is_err());

        // Password from an owner-only file, trailing newline removed.
        let password_path = directory.path().join("password");
        std::fs::write(&password_path, "s3cret \"with\" quotes\\\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&password_path, std::fs::Permissions::from_mode(0o600))
                .unwrap();
        }
        let tor = FakeTor::start(FakeTorConfig::password(b"s3cret \"with\" quotes\\")).await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        control
            .authenticate(&AuthChoice::Password(password_path.clone()))
            .await
            .unwrap();

        // A wrong password is refused by Tor and reported.
        std::fs::write(&password_path, "wrong\n").unwrap();
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        assert!(control
            .authenticate(&AuthChoice::Password(password_path.clone()))
            .await
            .is_err());

        // A group-readable password file is refused before anything is sent.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::write(&password_path, "s3cret \"with\" quotes\\\n").unwrap();
            std::fs::set_permissions(&password_path, std::fs::Permissions::from_mode(0o640))
                .unwrap();
            let before = tor.received().len();
            let mut control = TorControl::connect(tor.addr).await.unwrap();
            assert!(control
                .authenticate(&AuthChoice::Password(password_path.clone()))
                .await
                .is_err());
            assert_eq!(tor.received().len(), before);
        }
    }

    #[tokio::test]
    async fn control_add_onion_new_returns_a_validated_service_and_key() {
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        control.authenticate(&AuthChoice::Null).await.unwrap();
        let added = control
            .add_onion(KeyRequest::New, 9700, "127.0.0.1:19700".parse().unwrap())
            .await
            .unwrap();
        assert_eq!(added.service_id.as_string(), SYNTHETIC);
        assert_eq!(added.new_key.as_ref().unwrap().key_blob(), KEY_BLOB);
        assert!(tor
            .received()
            .contains(&"ADD_ONION NEW:ED25519-V3 Port=9700,127.0.0.1:19700\r\n".to_string()));

        let stored = sample_key();
        let again = control
            .add_onion(
                KeyRequest::Existing(&stored),
                9700,
                "127.0.0.1:19701".parse().unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(again.service_id.as_string(), SYNTHETIC);
        assert!(again.new_key.is_none());
        assert!(tor.received().contains(&format!(
            "ADD_ONION {KEY_BLOB} Port=9700,127.0.0.1:19701\r\n"
        )));

        control.del_onion(&again.service_id).await.unwrap();
        assert!(tor
            .received()
            .contains(&format!("DEL_ONION {SYNTHETIC}\r\n")));
    }

    #[tokio::test]
    async fn control_add_onion_refuses_untrusted_replies_and_removes_the_service() {
        // Invalid ServiceID (bad checksum).
        let tor = FakeTor::start(FakeTorConfig {
            new_service_id: "aaaqeayeaudaocajbifqydiob4ibceqtcqkrmfyydenbwha5dyp3iead".into(),
            ..FakeTorConfig::null()
        })
        .await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        control.authenticate(&AuthChoice::Null).await.unwrap();
        assert!(control
            .add_onion(KeyRequest::New, 9700, "127.0.0.1:19700".parse().unwrap())
            .await
            .is_err());

        // A private key that could inject a command is refused, and the
        // service Tor created is withdrawn.
        let tor = FakeTor::start(FakeTorConfig {
            new_key_blob: "ED25519-V3:abc DEL_ONION".into(),
            ..FakeTorConfig::null()
        })
        .await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        control.authenticate(&AuthChoice::Null).await.unwrap();
        assert!(control
            .add_onion(KeyRequest::New, 9700, "127.0.0.1:19700".parse().unwrap())
            .await
            .is_err());
        assert!(tor
            .received()
            .contains(&format!("DEL_ONION {SYNTHETIC}\r\n")));

        // The stored key yields another address: refuse and withdraw.
        let other = "uws2ljnfuws2ljnfuws2ljnfuws2ljnfuws2ljnfuws2ljnfuws4ypqd";
        let tor = FakeTor::start(FakeTorConfig {
            existing_service_id: other.into(),
            ..FakeTorConfig::null()
        })
        .await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        control.authenticate(&AuthChoice::Null).await.unwrap();
        let stored = sample_key();
        assert!(control
            .add_onion(
                KeyRequest::Existing(&stored),
                9700,
                "127.0.0.1:19700".parse().unwrap()
            )
            .await
            .is_err());
        assert!(tor.received().contains(&format!("DEL_ONION {other}\r\n")));

        // A Tor error is an error.
        let tor = FakeTor::start(FakeTorConfig {
            add_onion_error: Some("512 Invalid VIRTPORT/TARGET".into()),
            ..FakeTorConfig::null()
        })
        .await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        control.authenticate(&AuthChoice::Null).await.unwrap();
        assert!(matches!(
            control
                .add_onion(KeyRequest::New, 9700, "127.0.0.1:19700".parse().unwrap())
                .await,
            Err(OnionError::ControlRefused { code: 512, .. })
        ));
    }

    #[tokio::test]
    async fn control_add_onion_targets_only_loopback_and_a_real_port() {
        let tor = FakeTor::start(FakeTorConfig::null()).await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        control.authenticate(&AuthChoice::Null).await.unwrap();
        let before = tor.received().len();
        for (port, target) in [
            (9700, "192.168.1.2:19700"),
            (9700, "0.0.0.0:19700"),
            (0, "127.0.0.1:19700"),
            (9700, "127.0.0.1:0"),
        ] {
            assert!(control
                .add_onion(KeyRequest::New, port, target.parse().unwrap())
                .await
                .is_err());
        }
        assert_eq!(tor.received().len(), before);
    }

    #[tokio::test]
    async fn control_unauthenticated_add_onion_is_refused_by_tor() {
        let tor = FakeTor::start(FakeTorConfig::password(b"pw")).await;
        let mut control = TorControl::connect(tor.addr).await.unwrap();
        assert!(matches!(
            control
                .add_onion(KeyRequest::New, 9700, "127.0.0.1:19700".parse().unwrap())
                .await,
            Err(OnionError::ControlRefused { code: 514, .. })
        ));
    }

    #[tokio::test]
    async fn control_wait_closed_returns_when_tor_goes_away() {
        let server = ScriptedServer::start(vec![]).await;
        let mut control = TorControl::connect(server.addr).await.unwrap();
        server.close_all();
        let waited = tokio::time::timeout(Duration::from_secs(5), control.wait_closed()).await;
        assert!(waited.is_ok());
    }
}
