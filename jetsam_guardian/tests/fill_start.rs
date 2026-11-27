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

/// F-7 (external audit of the robots): a `<name>.tmp` planted as a symbolic link by whoever may write the state
/// directory is never written through: the file it names stays as it was, and the state written is a file of its own.
#[test]
fn f7_write_atomic_never_writes_through_a_planted_tmp_link() {
    if std::env::var_os("JG_WRITE_ATOMIC_CHILD").is_some() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let victim = dir.path().join("victim");
    std::fs::write(&victim, b"untouched").unwrap();
    let target = dir.path().join("state.json");
    std::os::unix::fs::symlink(&victim, dir.path().join("state.json.tmp")).unwrap();
    jetsam_guardian::write_atomic(&target, b"{\"new\":1}").unwrap();
    assert_eq!(std::fs::read(&victim).unwrap(), b"untouched", "the planted link's target is never written");
    assert!(!std::fs::symlink_metadata(&target).unwrap().file_type().is_symlink(), "the state is a file, not the planted link");
    assert_eq!(std::fs::read(&target).unwrap(), b"{\"new\":1}");
}

/// F-7 (decision D-3): the state directory of a robot holding a hot key (`jg-relay`, `jg-fill`) is created 0700; on
/// the public networks one that others may read or write (or a symbolic link, or another user's) does not start it
/// (whoever writes there may remove `HALT` or `gas.json`); on the private rehearsal it is not checked.
#[test]
fn d3_a_robots_state_dir_is_private_to_its_user_on_the_public_networks() {
    if std::env::var_os("JG_WRITE_ATOMIC_CHILD").is_some() {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let fresh = dir.path().join("a/state");
    jetsam_guardian::private_state_dir(&fresh, true).unwrap();
    assert_eq!(std::fs::metadata(&fresh).unwrap().permissions().mode() & 0o777, 0o700, "created 0700");
    jetsam_guardian::private_state_dir(&fresh, true).unwrap();
    let open = dir.path().join("open");
    std::fs::create_dir(&open).unwrap();
    for mode in [0o755, 0o750, 0o711, 0o770, 0o707] {
        std::fs::set_permissions(&open, std::fs::Permissions::from_mode(mode)).unwrap();
        let e = jetsam_guardian::private_state_dir(&open, true).unwrap_err().to_string();
        assert!(e.contains("0700"), "{mode:o}: {e}");
        assert!(jetsam_guardian::private_state_dir(&open, false).is_ok(), "{mode:o}: the rehearsal does not check it");
    }
    std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o700)).unwrap();
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(&open, &link).unwrap();
    let e = jetsam_guardian::private_state_dir(&link, true).unwrap_err().to_string();
    assert!(e.contains("symbolic link"), "{e}");
}

#[test]
fn a5_14_write_atomic_child() {
    let Some(target) = std::env::var_os("JG_WRITE_ATOMIC_CHILD") else { return };
    jetsam_guardian::write_atomic(std::path::Path::new(&target), b"{\"child\":1}").unwrap();
}

/// E-2 (audit 2026-11-10): `jg-fill` with its hot key (`[signer] key_file`) does not start when its memory cannot be
/// locked (a bounded `RLIMIT_MEMLOCK`, no `CAP_IPC_LOCK`): before the key is read and before any socket;
/// `--allow-unlocked-memory` lets it go on, with an ALERT. Without `prlimit`/`setpriv` on the host: skipped, said.
#[test]
fn jg_fill_holding_its_hot_key_does_not_start_with_its_memory_unlocked_unless_told() {
    if Command::new("setpriv").arg("--version").output().is_err() || Command::new("prlimit").arg("--version").output().is_err() {
        eprintln!("SKIPPED: no setpriv or prlimit on this host");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let key = dir.path().join("hot.key");
    let config = dir.path().join("fill.toml");
    let text = format!(
        "network = \"{}\"\nl1_chain_id = 31337\nl1_bridge = \"0x{}\"\nv1_7_height = 96\ndeposits_from_height = 128\n\
         l1_rpc = \"http://127.0.0.1:9\"\nhot_address = \"0x70997970C51812dc3A010C7d01b50e0d17dc79C8\"\n\
         refund_to = \"0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC\"\nwei_per_jtm = \"1000000000000000\"\nstate_dir = \"{}\"\npoll_secs = 1\n\
         [[nodes]]\nname = \"n1\"\nrpc = \"http://127.0.0.1:9\"\n[[nodes]]\nname = \"n2\"\nrpc = \"http://127.0.0.1:10\"\n\
         [params]\nshare_halt_pct = 100\nswitch_pause_secs = 300\n[egress]\nmode = \"clearnet\"\n[signer]\nkey_file = \"{}\"\n",
        hex::encode(jetsam_guardian::testkit::genesis().id),
        "ee".repeat(20),
        dir.path().join("state").display(),
        key.display()
    );
    std::fs::write(&config, text).unwrap();
    std::fs::set_permissions(&config, std::fs::Permissions::from_mode(0o600)).unwrap();
    let run = |args: &[&str]| {
        let out = Command::new("prlimit")
            .args(["--memlock=1048576:1048576", "setpriv", "--bounding-set=-ipc_lock", env!("CARGO_BIN_EXE_jg-fill")])
            .args(args)
            .arg("--config")
            .arg(&config)
            .output()
            .unwrap();
        (out.status.code(), String::from_utf8_lossy(&out.stderr).to_string())
    };
    let (code, err) = run(&[]);
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains("memory NOT locked") && err.contains("holds a hot key and its memory is not locked") && err.contains("jg-fill does not start"), "{err}");
    assert!(!err.contains("ALERT MEMORY_NOT_LOCKED"), "not started anyway: {err}");
    assert!(!err.contains("hot.key") && !err.contains("egress"), "refused before the key and before any socket: {err}");
    let (code, err) = run(&["--allow-unlocked-memory"]);
    assert_eq!(code, Some(1), "{err}");
    assert!(err.contains("ALERT MEMORY_NOT_LOCKED") && err.contains("--allow-unlocked-memory"), "{err}");
    assert!(!err.contains("its memory is not locked"), "past the gate: {err}");
}
