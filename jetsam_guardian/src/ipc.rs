// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The local channels between `jg-observe`, `jg-net` and `jg-sign`: Unix
//! stream sockets on the file system (they work across network namespaces:
//! `jg-sign` has NO network), one JSON request per line, one JSON answer per
//! line. On a guardian machine each socket's directory belongs to the one
//! account allowed to write to it (design §2.3); in the private rehearsal
//! every process runs as root (no account is created on the host).
//!
//! A4-2: each listening socket answers ONE uid, checked by `SO_PEERCRED` on
//! every connection before its request is read; the socket file is 0600 when
//! that uid is the server's own, 0660 otherwise (the client's account is then
//! a member of the server account's group).

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
    let listener = bind(path, own_uid())?;
    while !stop.load(Ordering::SeqCst) {
        if !serve_one(&listener, handler)? {
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    Ok(())
}

/// A listening socket and the one uid allowed to talk on it (A4-2).
pub struct Listener {
    inner: UnixListener,
    peer_uid: u32,
}

impl Listener {
    /// The uid this socket answers.
    pub fn peer_uid(&self) -> u32 {
        self.peer_uid
    }
}

/// This process's effective uid.
pub fn own_uid() -> u32 {
    // SAFETY: geteuid has no precondition and cannot fail.
    unsafe { libc::geteuid() }
}

/// The uid at the other end of `s` (`SO_PEERCRED`: the kernel's record of
/// the connecting process, not anything it sent).
pub fn peer_uid_of(s: &UnixStream) -> std::io::Result<u32> {
    use std::os::unix::io::AsRawFd;
    let mut cred = libc::ucred { pid: 0, uid: 0, gid: 0 };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: `cred` and `len` are valid for writes of the sizes given; the
    // descriptor belongs to `s` for the whole call.
    let r = unsafe {
        libc::getsockopt(s.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, (&mut cred as *mut libc::ucred).cast(), &mut len)
    };
    if r != 0 {
        return Err(std::io::Error::last_os_error());
    }
    if len as usize != std::mem::size_of::<libc::ucred>() {
        return Err(std::io::Error::other("SO_PEERCRED: short answer"));
    }
    Ok(cred.uid)
}

/// Bind a non-blocking listener at `path` (the old socket file replaced)
/// for peers of uid `peer_uid`: the file is 0600 when `peer_uid` is this
/// process's uid, 0660 otherwise.
pub fn bind(path: &Path, peer_uid: u32) -> anyhow::Result<Listener> {
    use std::os::unix::fs::PermissionsExt;
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let listener = UnixListener::bind(path)?;
    // (between bind and chmod the file has the umask's mode: SO_PEERCRED
    // below refuses any other uid all the same)
    let mode = if peer_uid == own_uid() { 0o600 } else { 0o660 };
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))?;
    listener.set_nonblocking(true)?;
    Ok(Listener { inner: listener, peer_uid })
}

/// Serve at most one pending connection; `false` when none was waiting.
pub fn serve_one(listener: &Listener, handler: &mut dyn FnMut(Value) -> Value) -> anyhow::Result<bool> {
    match listener.inner.accept() {
        Ok((mut s, _)) => {
            s.set_nonblocking(false)?;
            // A4-2: the request of any other uid is never read
            if peer_uid_of(&s).ok() != Some(listener.peer_uid) {
                let _ = s.write_all(b"{\"error\":\"PEER_REFUSED\"}\n");
                return Ok(true);
            }
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

    // ---- A4-2: who may talk on a socket ----

    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    fn mode(path: &Path) -> u32 {
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    /// A refused peer reads `PEER_REFUSED`, or finds the socket closed when the
    /// server refused before the request was written: never the handler's answer.
    fn assert_refused(r: anyhow::Result<Value>) {
        match r {
            Ok(a) => assert_eq!(a, json!({"error": "PEER_REFUSED"})),
            Err(e) => eprintln!("refused before the request was written ({e})"),
        }
    }

    /// Serve `listener` in a thread until `stop`, counting the handled requests.
    fn spawn_server(listener: Listener, stop: Arc<AtomicBool>, handled: Arc<AtomicUsize>) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                let h = handled.clone();
                let served = serve_one(&listener, &mut |v: Value| {
                    h.fetch_add(1, Ordering::SeqCst);
                    json!({"echo": v})
                })
                .unwrap();
                if !served {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
        })
    }

    #[test]
    fn socket_is_0600_for_its_own_uid_and_0660_for_another() {
        let dir = tempfile::tempdir().unwrap();
        let own = dir.path().join("own.sock");
        let l = bind(&own, own_uid()).unwrap();
        assert_eq!(mode(&own), 0o600);
        assert_eq!(l.peer_uid(), own_uid());
        let other = dir.path().join("other.sock");
        let peer = own_uid().wrapping_add(4_242);
        let l = bind(&other, peer).unwrap();
        assert_eq!(mode(&other), 0o660, "another account reaches it through this account's group only");
        assert_eq!(l.peer_uid(), peer);
        // a rebind replaces the file and sets the mode again
        std::fs::set_permissions(&own, std::fs::Permissions::from_mode(0o666)).unwrap();
        let _l = bind(&own, own_uid()).unwrap();
        assert_eq!(mode(&own), 0o600);
    }

    #[test]
    fn a_peer_of_another_uid_is_refused_before_its_request_is_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.sock");
        let stop = Arc::new(AtomicBool::new(false));
        let handled = Arc::new(AtomicUsize::new(0));
        // the socket answers uid own+4242 only; this test runs as `own`
        let server = spawn_server(bind(&path, own_uid().wrapping_add(4_242)).unwrap(), stop.clone(), handled.clone());
        assert_refused(request(&path, &json!({"op": "sign"}), Duration::from_secs(5)));
        assert_eq!(handled.load(Ordering::SeqCst), 0, "the handler never saw the request");
        stop.store(true, Ordering::SeqCst);
        server.join().unwrap();
        // the expected uid is served
        let stop = Arc::new(AtomicBool::new(false));
        let server = spawn_server(bind(&path, own_uid()).unwrap(), stop.clone(), handled.clone());
        let a = request(&path, &json!({"op": "status"}), Duration::from_secs(5)).unwrap();
        assert_eq!(a, json!({"echo": {"op": "status"}}));
        assert_eq!(handled.load(Ordering::SeqCst), 1);
        stop.store(true, Ordering::SeqCst);
        server.join().unwrap();
    }

    #[test]
    fn so_peercred_reads_the_real_uid_of_another_account() {
        // root only: a client under uid 65534, gid 65533 (setpriv) on a socket meant
        // for uid 65534 is served, root on the same socket is refused
        if own_uid() != 0 {
            eprintln!("SKIPPED: needs root (setpriv to another uid)");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = dir.path().join("s.sock");
        let stop = Arc::new(AtomicBool::new(false));
        let handled = Arc::new(AtomicUsize::new(0));
        let listener = bind(&path, 65_534).unwrap();
        // (the group route of 0660 is not what is tested here: open the file to everyone)
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666)).unwrap();
        let server = spawn_server(listener, stop.clone(), handled.clone());
        let client = r#"import socket,sys
s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); s.sendall(b'{"op":"x"}\n')
print(s.makefile().readline().strip())"#;
        let out = std::process::Command::new("setpriv")
            .args(["--reuid=65534", "--regid=65533", "--clear-groups", "python3", "-I", "-c", client])
            .arg(&path)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let answer: Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(answer, json!({"echo": {"op": "x"}}), "uid 65534 is the expected peer");
        assert_refused(request(&path, &json!({"op": "x"}), Duration::from_secs(5)));
        assert_eq!(handled.load(Ordering::SeqCst), 1, "root is not served");
        stop.store(true, Ordering::SeqCst);
        server.join().unwrap();
    }

    #[test]
    fn peer_uid_of_a_socket_pair_is_this_process() {
        let (a, _b) = UnixStream::pair().unwrap();
        assert_eq!(peer_uid_of(&a).unwrap(), own_uid());
    }
}
