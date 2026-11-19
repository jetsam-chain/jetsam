// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! M-12 (external audit of the robots): `jg-keys` writes its secrets 0600 whatever the umask, and seals a machine key
//! with the TEST cost only when the operator says, in words, that it is for the private rehearsal.

use std::os::unix::fs::PermissionsExt;
use std::process::{Command, Stdio};

fn jg_keys(args: &[&str], dir: &std::path::Path) -> std::process::Output {
    let pass = dir.join("TEST-passphrase");
    std::fs::write(&pass, "TEST passphrase\n").unwrap();
    // a permissive umask: the files must not follow it
    Command::new("/bin/sh")
        .arg("-c")
        .arg("umask 022; exec \"$@\"")
        .arg("sh")
        .arg(env!("CARGO_BIN_EXE_jg-keys"))
        .args(args)
        .stdin(std::fs::File::open(&pass).unwrap())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .unwrap()
}

fn mode(p: &std::path::Path) -> u32 {
    std::fs::metadata(p).unwrap().permissions().mode() & 0o777
}

#[test]
fn m12_the_feed_key_and_the_sealed_machine_key_are_0600() {
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let s = state.to_str().unwrap();
    let out = jg_keys(&["feed-init", "--state", s], dir.path());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(mode(&state.join("net/feed.key")), 0o600);
    let out = jg_keys(&["machine-init", "--state", s, "--generation", "1", "--passphrase-fd", "0", "--test-kdf", "--private-rehearsal"], dir.path());
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(mode(&state.join("sign/machine-g1.key.sealed")), 0o600);
    assert!(String::from_utf8_lossy(&out.stdout).contains("TEST"), "the output says the key is a TEST one");
}

#[test]
fn m12_the_test_kdf_needs_the_private_rehearsal_said() {
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let out = jg_keys(&["machine-init", "--state", state.to_str().unwrap(), "--generation", "1", "--passphrase-fd", "0", "--test-kdf"], dir.path());
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--private-rehearsal"), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!state.join("sign/machine-g1.key.sealed").exists(), "nothing written");
}
