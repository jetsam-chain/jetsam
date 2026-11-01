// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The local channels between `jg-observe`, `jg-net` and `jg-sign`: Unix
//! stream sockets on the file system (they work across network namespaces:
//! `jg-sign` has NO network), one JSON request per line, one JSON answer per
//! line. On a guardian machine each socket's directory belongs to the one
//! account allowed to write to it (design §2.3); in the private rehearsal
//! every process runs as root (no account is created on the host).

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::Value;

/// One request to the socket at `path`, its answer.
pub fn request(path: &Path, req: &Value, timeout: Duration) -> anyhow::Result<Value> {
    let mut s = UnixStream::connect(path)?;
    s.set_read_timeout(Some(timeout))?;
    s.set_write_timeout(Some(timeout))?;
    let mut line = serde_json::to_vec(req)?;
    line.push(b'\n');
    s.write_all(&line)?;
    let mut answer = String::new();
    BufReader::new(&s).read_line(&mut answer)?;
    anyhow::ensure!(!answer.is_empty(), "no answer from {}", path.display());
    Ok(serde_json::from_str(answer.trim_end())?)
}

/// Serve `path` until `stop` is set: each connection carries one line, the
/// handler's value is written back. The socket file is replaced at start.
pub fn serve(path: &Path, stop: &AtomicBool, handler: &mut dyn FnMut(Value) -> Value) -> anyhow::Result<()> {
    let listener = bind(path)?;
    while !stop.load(Ordering::SeqCst) {
        if !serve_one(&listener, handler)? {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    Ok(())
}

/// Bind a non-blocking listener at `path` (the old socket file replaced).
pub fn bind(path: &Path) -> anyhow::Result<UnixListener> {
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let listener = UnixListener::bind(path)?;
    listener.set_nonblocking(true)?;
    Ok(listener)
}

/// Serve at most one pending connection; `false` when none was waiting.
pub fn serve_one(listener: &UnixListener, handler: &mut dyn FnMut(Value) -> Value) -> anyhow::Result<bool> {
    match listener.accept() {
        Ok((mut s, _)) => {
            s.set_nonblocking(false)?;
            s.set_read_timeout(Some(Duration::from_secs(10)))?;
            let mut line = String::new();
            if BufReader::new(&s).read_line(&mut line).is_ok() && !line.trim().is_empty() {
                let answer = match serde_json::from_str::<Value>(line.trim_end()) {
                    Ok(v) => handler(v),
                    Err(e) => serde_json::json!({"error": format!("bad request: {e}")}),
                };
                let mut out = serde_json::to_vec(&answer)?;
                out.push(b'\n');
                let _ = s.write_all(&out);
            }
            Ok(true)
        }
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(false),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ipc_round_trip_and_stop() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.sock");
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let p2 = path.clone();
        let s2 = stop.clone();
        let server = std::thread::spawn(move || {
            serve(&p2, &s2, &mut |v: Value| json!({"echo": v})).unwrap();
        });
        for _ in 0..100 {
            if path.exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let a = request(&path, &json!({"op": "x", "n": 1}), Duration::from_secs(5)).unwrap();
        assert_eq!(a, json!({"echo": {"op": "x", "n": 1}}));
        let b = request(&path, &json!([1, 2]), Duration::from_secs(5)).unwrap();
        assert_eq!(b["echo"], json!([1, 2]));
        stop.store(true, Ordering::SeqCst);
        server.join().unwrap();
    }
}
