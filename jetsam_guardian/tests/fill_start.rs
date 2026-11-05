// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A5-15: `jg-fill` hardens itself before it reads anything (no core dump, not dumpable, memory locked when
//! that is safe), and refuses a `fill.toml` that is not private to its user, before parsing it; the shared
//! `jetsam_guardian::harden` in a child process.

use std::os::unix::fs::PermissionsExt;
use std::process::Command;

fn start(config: &std::path::Path) -> (Option<i32>, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_jg-fill")).arg("--config").arg(config).output().unwrap();
    (out.status.code(), String::from_utf8_lossy(&out.stderr).to_string())
}

#[test]
fn jg_fill_hardens_itself_then_refuses_a_fill_toml_others_can_read() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("fill.toml");
    std::fs::write(&config, "not = \"a configuration\"\n").unwrap();
    std::fs::set_permissions(&config, std::fs::Permissions::from_mode(0o644)).unwrap();
    let (code, err) = start(&config);
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains("hardened: dumpable off, core dumps off"), "{err}");
    assert!(err.contains("0600") && err.contains("does not start"), "{err}");
    assert!(err.find("hardened").unwrap() < err.find("0600").unwrap(), "hardened first: {err}");
    // private: the mode passes, the parse refuses it (unknown field)
    std::fs::set_permissions(&config, std::fs::Permissions::from_mode(0o600)).unwrap();
    let (code, err) = start(&config);
    assert_eq!(code, Some(1), "{err}");
    assert!(!err.contains("0600") && err.contains("unknown field"), "{err}");
}

/// The hardening itself, in a child process (this test binary again, `harden_in_a_child_process` only; never a
/// unit test of the library, see `journal_tests.rs`), so
/// that the other tests never run with locked memory or without core dumps.
#[test]
fn harden_turns_core_dumps_and_ptrace_off_and_locks_memory() {
    if std::env::var_os("JG_HARDEN_CHILD").is_some() {
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "harden_in_a_child_process", "--nocapture", "--test-threads", "1"])
        .env("JG_HARDEN_CHILD", "1")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("CHILD dumpable=0 core=0/0"), "{text}");
    // the test host runs as root (CAP_IPC_LOCK): memory is locked
    if jetsam_guardian::harden::may_lock_all(0, cap_eff()) {
        assert!(text.contains("locked=ok"), "{text}");
        let vmlck: u64 = text.split("vmlck_kb=").nth(1).and_then(|s| s.split_whitespace().next()).and_then(|s| s.parse().ok()).unwrap();
        assert!(vmlck > 0, "{text}");
    }
}

fn cap_eff() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|l| l.starts_with("CapEff:")).unwrap();
    u64::from_str_radix(line.split_whitespace().nth(1).unwrap(), 16).unwrap()
}

#[test]
fn harden_in_a_child_process() {
    if std::env::var_os("JG_HARDEN_CHILD").is_none() {
        return;
    }
    let h = jetsam_guardian::harden::harden();
    // SAFETY: plain queries of this process.
    let dumpable = unsafe { libc::prctl(libc::PR_GET_DUMPABLE, 0, 0, 0, 0) };
    let mut core = libc::rlimit { rlim_cur: 1, rlim_max: 1 };
    unsafe { libc::getrlimit(libc::RLIMIT_CORE, &mut core) };
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let vmlck = status.lines().find(|l| l.starts_with("VmLck:")).and_then(|l| l.split_whitespace().nth(1)).unwrap_or("0").to_string();
    println!(
        "CHILD dumpable={dumpable} core={}/{} locked={} vmlck_kb={vmlck} {h:?}",
        core.rlim_cur,
        core.rlim_max,
        if h.memory_locked.is_ok() { "ok" } else { "no" }
    );
    assert!(h.dumpable_off && h.core_off);
}
