//! The launch scripts (design §8.1, §8.2): allow-list, PID file, verified stop.

use std::path::{Path, PathBuf};
use std::process::Command;

fn ops(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("ops").join(name)
}

fn sha(p: &Path) -> String {
    let out = Command::new("sha256sum").arg(p).output().unwrap();
    String::from_utf8(out.stdout).unwrap().split_whitespace().next().unwrap().to_string()
}

/// A state dir and a bin dir whose `jg-observe` is the TEST sleeper.
fn setup(allow: bool) -> (tempfile::TempDir, PathBuf, PathBuf, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let bin = root.path().join("bin");
    let state = root.path().join("state");
    std::fs::create_dir_all(&bin).unwrap();
    std::fs::create_dir_all(&state).unwrap();
    std::fs::copy(env!("CARGO_BIN_EXE_jg-test-sleep"), bin.join("jg-observe")).unwrap();
    let config = root.path().join("guardian.toml");
    std::fs::write(&config, "index = 1\n").unwrap();
    let mut lines = vec![sha(&config)];
    if allow {
        lines.push(sha(&bin.join("jg-observe")));
    }
    std::fs::write(state.join("allowed.sha256"), lines.join("\n") + "\n").unwrap();
    (root, bin, state, config)
}

fn start(bin: &Path, state: &Path, config: &Path) -> std::process::Output {
    Command::new(ops("jg-start"))
        .args(["observe", "--state"]).arg(state).arg("--config").arg(config).arg("--bin-dir").arg(bin)
        .output()
        .unwrap()
}

fn pid_of(state: &Path) -> u32 {
    std::fs::read_to_string(state.join("run/observe.pid")).unwrap().split_whitespace().next().unwrap().parse().unwrap()
}

fn alive(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}/stat")).exists()
        && !std::fs::read_to_string(format!("/proc/{pid}/stat")).unwrap_or_default().contains(") Z ")
}

#[test]
fn start_refuses_binary_not_in_allowlist() {
    let (_root, bin, state, config) = setup(false);
    let out = start(&bin, &state, &config);
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!state.join("run/observe.pid").exists());
}

#[test]
fn stop_kills_exact_pid_after_starttime_check() {
    let (_root, bin, state, config) = setup(true);
    let out = start(&bin, &state, &config);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let pid = pid_of(&state);
    assert!(alive(pid));
    // a second instance is refused
    assert_eq!(start(&bin, &state, &config).status.code(), Some(4));
    let stop = Command::new(ops("jg-stop")).args(["observe", "--state"]).arg(&state).output().unwrap();
    assert!(stop.status.success(), "{}", String::from_utf8_lossy(&stop.stderr));
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(!alive(pid));
    assert!(!state.join("run/observe.pid").exists());
}

#[test]
fn stop_refuses_reused_pid() {
    let (_root, bin, state, config) = setup(true);
    assert!(start(&bin, &state, &config).status.success());
    let pid = pid_of(&state);
    let pidf = state.join("run/observe.pid");
    let good = std::fs::read_to_string(&pidf).unwrap();
    let mut f: Vec<String> = good.split_whitespace().map(str::to_string).collect();
    f[1] = (f[1].parse::<u64>().unwrap() + 1).to_string(); // another start time: a reused PID
    std::fs::write(&pidf, f.join(" ") + "\n").unwrap();
    let stop = Command::new(ops("jg-stop")).args(["observe", "--state"]).arg(&state).output().unwrap();
    assert_eq!(stop.status.code(), Some(3));
    assert!(alive(pid), "a reused PID is never signalled");
    std::fs::write(&pidf, good).unwrap();
    assert!(Command::new(ops("jg-stop")).args(["observe", "--state"]).arg(&state).status().unwrap().success());
}

#[test]
fn no_pattern_kill_in_ops_scripts() {
    for name in ["jg-start", "jg-stop"] {
        let text = std::fs::read_to_string(ops(name)).unwrap();
        for line in text.lines().filter(|l| !l.trim_start().starts_with('#')) {
            for bad in ["pkill", "killall", "pgrep", "kill -9 -", "kill 0"] {
                assert!(!line.contains(bad), "{name}: {line}");
            }
        }
    }
}
