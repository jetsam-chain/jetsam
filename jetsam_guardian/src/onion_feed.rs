// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! C-2 (a) (audit 2026-11-10): the guardians' feeds over Tor, PULLED, never pushed.
//!
//! A publication point used to be a directory synchronised by hand between machines (rsync, git, HTTPS from
//! the guardian's machine): the host it was pushed to learnt the guardian's address together with its signed
//! `g<i>` line every minute. Now each guardian SERVES its own feed read only on its loopback — a Tor onion
//! service (`HiddenServicePort` in its torrc) points at it, so no connection ever leaves the guardian's machine
//! for its publication — and the others PULL it from that `.onion` through Tor's SOCKS port ([`pull_feed`]),
//! keeping only the lines that verify under its feed key, into their own local publication point, where
//! `jg-net` reads its sources as before.
//!
//! The server answers `GET /feeds/g<i>/<YYYY-MM-DD>.jsonl` with that file's bytes and nothing else: no other
//! path (no listing, no `..`, no other file of the directory), no other method, no write.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::feed::{feed_path, verify_line};
use crate::rpc::Egress;

/// Bound on a request head read by the server.
const MAX_REQUEST_HEAD: usize = 4_096;

/// The guardian index and the day of a request path `/feeds/g<i>/<YYYY-MM-DD>.jsonl`; `None` for any other.
pub fn feed_request_path(path: &str) -> Option<(u32, String)> {
    let rest = path.strip_prefix("/feeds/g")?;
    let (index, file) = rest.split_once('/')?;
    if index.is_empty() || index.starts_with('0') || !index.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let day = file.strip_suffix(".jsonl")?;
    let b = day.as_bytes();
    let shape = b.len() == 10
        && b.iter().enumerate().all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() });
    if !shape {
        return None;
    }
    Some((index.parse().ok()?, day.to_string()))
}

/// Serve the feeds under `root` read only on `listen` (a loopback IP address and port, `:0` for any free one),
/// one thread for the life of the process; the address bound. Off the loopback: refused, nothing bound.
pub fn serve(listen: &str, root: PathBuf) -> anyhow::Result<SocketAddr> {
    let addr: SocketAddr = listen.parse().map_err(|_| anyhow::anyhow!("feed_listen {listen:?}: an IP address and port"))?;
    anyhow::ensure!(addr.ip().is_loopback(), "feed_listen {listen}: a loopback address only (the onion service points at it)");
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let _ = answer(stream, &root);
        }
    });
    Ok(bound)
}

/// One request on `s`: its head read (bounded), then one answer, then the connection closed.
fn answer(mut s: TcpStream, root: &Path) -> std::io::Result<()> {
    s.set_read_timeout(Some(Duration::from_secs(10)))?;
    s.set_write_timeout(Some(Duration::from_secs(10)))?;
    let mut head = Vec::new();
    let mut buf = [0u8; 1_024];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") && head.len() < MAX_REQUEST_HEAD {
        let n = s.read(&mut buf)?;
        if n == 0 {
            break;
        }
        head.extend_from_slice(&buf[..n]);
    }
    let line = String::from_utf8_lossy(&head);
    let mut parts = line.lines().next().unwrap_or("").split(' ');
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let reply = |s: &mut TcpStream, status: &str, body: &[u8]| -> std::io::Result<()> {
        write!(s, "HTTP/1.1 {status}\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())?;
        s.write_all(body)
    };
    if method != "GET" {
        return reply(&mut s, "405 Method Not Allowed", b"");
    }
    match feed_request_path(path).and_then(|(g, day)| std::fs::read(feed_path(root, g, &day)).ok()) {
        Some(bytes) => reply(&mut s, "200 OK", &bytes),
        None => reply(&mut s, "404 Not Found", b""),
    }
}

/// Pull guardian `g`'s feed of `day` from `base` (its server: `http://<56>.onion[:port]`, through Tor by
/// `egress`) and keep the lines that verify under `g`'s own key in `keys`, in their order; written atomically
/// under `point` (`feeds/g<g>/<day>.jsonl`) when they are more than the local copy holds (a shorter answer never
/// shrinks it). Returns how many lines verified.
pub fn pull_feed(
    egress: &Egress,
    base: &str,
    g: u32,
    day: &str,
    keys: &HashMap<u32, [u8; 32]>,
    point: &Path,
    timeout: Duration,
) -> anyhow::Result<usize> {
    anyhow::ensure!(keys.contains_key(&g), "g{g}: no feed key configured");
    let url = format!("{}/feeds/g{g}/{day}.jsonl", base.trim_end_matches('/'));
    let body = crate::rpc::get_bytes_via(egress, &url, timeout)?;
    let text = String::from_utf8_lossy(&body);
    let mut kept = String::new();
    let mut n = 0;
    for line in text.lines() {
        if matches!(verify_line(line, keys), Ok((gi, _)) if gi == g) {
            kept.push_str(line);
            kept.push('\n');
            n += 1;
        }
    }
    let path = feed_path(point, g, day);
    let local = std::fs::read_to_string(&path).map(|t| t.lines().count()).unwrap_or(0);
    if n > local {
        std::fs::create_dir_all(path.parent().expect("a feed path has a parent"))?;
        crate::write_atomic(&path, kept.as_bytes())?;
    }
    Ok(n)
}

#[cfg(test)]
#[path = "onion_feed_tests.rs"]
mod tests;
