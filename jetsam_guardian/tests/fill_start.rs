// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A5-15: `jg-fill` hardens itself before it reads anything (no core dump, not dumpable, memory locked when
//! that is safe), and refuses a `fill.toml` that is not private to its user, before parsing it; the shared
//! `jetsam_guardian::harden` in a child process. A5-14: `jetsam_guardian::write_atomic`'s system calls, under strace
//! in a child process.

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

/// A5-14: `write_atomic` syncs the file before its rename and the directory after it (the order of the system calls,
/// read with strace in a child process: this test binary again, `a5_14_write_atomic_child` only; never a unit test
/// of the library, see `journal_tests.rs`). Without strace on the host: skipped, said.
#[test]
fn a5_14_write_atomic_syncs_the_file_then_renames_it_then_syncs_the_directory() {
    if std::env::var_os("JG_WRITE_ATOMIC_CHILD").is_some() {
        return;
    }
    if std::process::Command::new("strace").arg("-V").output().is_err() {
        eprintln!("SKIPPED: no strace on this host");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let real = std::fs::canonicalize(dir.path()).unwrap();
    let (trace, target) = (real.join("trace"), real.join("state.json"));
    let out = std::process::Command::new("strace")
        .args(["-f", "-y", "-qq", "-e", "trace=fsync,fdatasync,rename,renameat,renameat2", "-o"])
        .arg(&trace)
        .arg(std::env::current_exe().unwrap())
        .args(["--exact", "a5_14_write_atomic_child", "--test-threads", "1"])
        .env("JG_WRITE_ATOMIC_CHILD", &target)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    assert_eq!(std::fs::read_to_string(&target).unwrap(), "{\"child\":1}");
    let text = std::fs::read_to_string(&trace).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let tmp = format!("{}.tmp", target.display());
    let at = |what: &dyn Fn(&str) -> bool| lines.iter().position(|l| what(l));
    let file_synced = at(&|l: &str| (l.contains("fsync(") || l.contains("fdatasync(")) && l.contains(&format!("<{tmp}>)")));
    let renamed = at(&|l: &str| l.contains("rename") && l.contains(&format!("\"{tmp}\"")) && l.contains(&format!("\"{}\"", target.display())));
    let dir_synced = at(&|l: &str| l.contains("fsync(") && l.contains(&format!("<{}>)", real.display())));
    let (Some(f), Some(r), Some(d)) = (file_synced, renamed, dir_synced) else { panic!("file synced {file_synced:?}, renamed {renamed:?}, directory synced {dir_synced:?}:\n{text}") };
    assert!(f < r && r < d, "the file synced, then renamed, then the directory synced:\n{text}");
}

#[test]
fn a5_14_write_atomic_child() {
    let Some(target) = std::env::var_os("JG_WRITE_ATOMIC_CHILD") else { return };
    jetsam_guardian::write_atomic(std::path::Path::new(&target), b"{\"child\":1}").unwrap();
}
