// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The relayer's prover command (`relayer::CommandProver`): it spawns a process, so it is tested here and
//! never from the library's unit tests (`tests/fork_and_journal.rs`). The command gets the program's input
//! file, must write the proof, may write the public values it proved (checked), runs under a timeout, and
//! never through a shell (its arguments are substituted, not interpreted).

use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::Duration;

use jetsam_guardian::checkpoint::{Checkpoint, Mn};
use jetsam_guardian::relayer::{CommandProver, ProofRequest, Prover};

fn request() -> ProofRequest {
    let cp = Checkpoint {
        height: 40,
        cum_work: [1u8; 32],
        headers_digest: [2u8; 32],
        deposit_root: [3u8; 32],
        deposit_count: 0,
        anchor_root: [4u8; 32],
        locked_height: 32,
        tip_block_id: [5u8; 32],
        locked_block_id: [6u8; 32],
        locked_cum_work: [7u8; 32],
    };
    ProofRequest {
        parent_id: [8u8; 32],
        parent: cp,
        fork_depth: 0,
        anchor_base: 11,
        next: cp,
        mn: Mn::default(),
        public_values: vec![0x5a; 608],
        headers_from: 0,
        headers: Vec::new(),
        abandoned: Vec::new(),
        deposit_blocks: Vec::new(),
        deposit_frontier: [[0u8; 32]; 32],
        anchor_frontier: [[0u8; 32]; 32],
    }
}

fn encode(r: &ProofRequest) -> anyhow::Result<Vec<u8>> {
    let mut v = b"INPUT:".to_vec();
    v.extend_from_slice(&r.public_values[..4]);
    Ok(v)
}

fn script(dir: &Path, name: &str, body: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o700)).unwrap();
    p.to_string_lossy().into_owned()
}

fn prover(dir: &Path, cmd: String, timeout: Duration) -> CommandProver {
    CommandProver { argv: vec![cmd, "{input}".into(), "{proof}".into(), "{public_values}".into()], work_dir: dir.join("work"), timeout, encode }
}

#[test]
fn the_command_gets_the_input_and_its_proof_is_returned() {
    let dir = tempfile::tempdir().unwrap();
    let expect = dir.path().join("pv");
    std::fs::write(&expect, vec![0x5a; 608]).unwrap();
    let seen = dir.path().join("seen");
    let cmd = script(dir.path(), "ok", &format!("cp \"$1\" {s}; printf PROOF > \"$2\"; cp {e} \"$3\"", s = seen.display(), e = expect.display()));
    let proof = prover(dir.path(), cmd, Duration::from_secs(30)).prove(&request()).unwrap();
    assert_eq!(proof, b"PROOF");
    assert_eq!(std::fs::read(&seen).unwrap(), encode(&request()).unwrap(), "the program's input, as the encoder made it");
    // the public values are optional
    let cmd = script(dir.path(), "nopv", "printf P2 > \"$2\"");
    assert_eq!(prover(dir.path(), cmd, Duration::from_secs(30)).prove(&request()).unwrap(), b"P2");
}

#[test]
fn a_proof_of_other_public_values_a_failure_an_empty_proof_or_a_timeout_is_no_proof() {
    let dir = tempfile::tempdir().unwrap();
    let other = script(dir.path(), "other", "printf PROOF > \"$2\"; printf OTHER > \"$3\"");
    let e = prover(dir.path(), other, Duration::from_secs(30)).prove(&request()).unwrap_err().to_string();
    assert!(e.contains("other public values"), "{e}");
    let fail = script(dir.path(), "fail", "exit 3");
    let e = prover(dir.path(), fail, Duration::from_secs(30)).prove(&request()).unwrap_err().to_string();
    assert!(e.contains("exited"), "{e}");
    let empty = script(dir.path(), "empty", ": > \"$2\"");
    let e = prover(dir.path(), empty, Duration::from_secs(30)).prove(&request()).unwrap_err().to_string();
    assert!(e.contains("empty proof"), "{e}");
    let marker = dir.path().join("still-ran");
    let slow = script(dir.path(), "slow", &format!("sleep 4; touch {}", marker.display()));
    let start = std::time::Instant::now();
    let e = prover(dir.path(), slow, Duration::from_secs(1)).prove(&request()).unwrap_err().to_string();
    assert!(e.contains("timed out"), "{e}");
    assert!(start.elapsed() < Duration::from_secs(3));
    std::thread::sleep(Duration::from_secs(5));
    assert!(!marker.exists(), "the timed-out prover was killed");
}

#[test]
fn arguments_are_substituted_never_interpreted_by_a_shell() {
    let dir = tempfile::tempdir().unwrap();
    let args = dir.path().join("args");
    let cmd = script(dir.path(), "echoargs", &format!("printf '%s\\n' \"$@\" > {}; printf P > \"$2\"", args.display()));
    let mut p = prover(dir.path(), cmd, Duration::from_secs(30));
    p.argv.push("$(touch /tmp/never-made-by-jg-relay-test); `id`".into());
    p.prove(&request()).unwrap();
    let text = std::fs::read_to_string(&args).unwrap();
    assert!(text.contains("$(touch /tmp/never-made-by-jg-relay-test); `id`"), "{text}");
    assert!(!Path::new("/tmp/never-made-by-jg-relay-test").exists());
}
