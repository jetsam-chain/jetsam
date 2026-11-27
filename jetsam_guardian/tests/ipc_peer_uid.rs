// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A4-2, the one IPC test that needs another account: it spawns a client process (`setpriv`), so it lives
//! in a test binary of its own and never with the library's unit tests, where a fork breaks the signer
//! journal's lock (`tests/fork_and_journal.rs`).

use std::os::unix::fs::PermissionsExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use jetsam_guardian::ipc::{bind, own_uid, request, serve_one, Listener};

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
