// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The basectl backend: the exact command lines, the parsing of basectl's JSON (its shapes in base/base v1.5.4,
//! `crates/infra/basectl/src/commands/proofs.rs`), the bounded run of a command, and the whole backend driven against a
//! stand-in `basectl` script.

use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};

use super::*;
use crate::basezk::{BackendConfig, BackendKind, Delivery, DeliveryMode, Poll, ZkBackend};

const GAME: [u8; 20] = [0x6a; 20];
const PROVER: [u8; 20] = [0xee; 20];
const FACTORY: [u8; 20] = [0xd6; 20];

fn cfg(kind: BackendKind, delivery: DeliveryMode, basectl: &str) -> BackendConfig {
    BackendConfig {
        kind,
        basectl: basectl.into(),
        config_name: "sepolia".into(),
        prover_rpc: "http://127.0.0.1:9000".into(),
        basectl_l1_rpc: None,
        delivery,
        proof_bytes_command: vec!["/opt/export-proof".into(), "{session}".into(), "{game}".into()],
        timeout_secs: 30,
        submit_timeout_secs: 900,
        retry_base_secs: 60,
        retry_max_secs: 3_600,
        alert_after_attempts: 5,
    }
}

fn backend(kind: BackendKind, delivery: DeliveryMode, key: Option<&str>) -> BasectlBackend {
    BasectlBackend::new(cfg(kind, delivery, "/usr/local/bin/basectl"), FACTORY, "http://127.0.0.1:8545", key.map(Into::into))
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

#[test]
fn propose_asks_exactly_one_game_bound_to_our_address_without_a_prompt() {
    let b = backend(BackendKind::DryRun, DeliveryMode::ProofBytesCommand, None);
    let game = format!("0x{}", hex::encode(GAME));
    let prover = format!("0x{}", hex::encode(PROVER));
    let factory = format!("0x{}", hex::encode(FACTORY));
    assert_eq!(
        b.propose_argv(&GAME, &PROVER, false),
        s(&[
            "/usr/local/bin/basectl", "-c", "sepolia", "proofs", "propose", &game, "--prover-address", &prover, "--zk-backend", "dry-run",
            "--prover-rpc", "http://127.0.0.1:9000", "--factory", &factory, "--l1-rpc", "http://127.0.0.1:8545", "--yes", "--json",
        ])
    );
    let b = backend(BackendKind::Network, DeliveryMode::ProofBytesCommand, Some("/k"));
    let argv = b.propose_argv(&GAME, &PROVER, true);
    assert_eq!(argv[9], "network");
    assert!(argv.contains(&"--retry-failed".to_string()), "a failed session is paid again only when asked");
    assert!(!b.propose_argv(&GAME, &PROVER, false).contains(&"--retry-failed".to_string()));
    assert_eq!(backend(BackendKind::Cluster, DeliveryMode::BasectlSubmit, Some("/k")).propose_argv(&GAME, &PROVER, false)[9], "cluster");
}

#[test]
fn status_and_submit_command_lines() {
    let b = backend(BackendKind::Network, DeliveryMode::BasectlSubmit, Some("/etc/jg/key"));
    assert_eq!(b.status_argv("sess-1"), s(&["/usr/local/bin/basectl", "-c", "sepolia", "proofs", "status", "sess-1", "--prover-rpc", "http://127.0.0.1:9000", "--json"]));
    let game = format!("0x{}", hex::encode(GAME));
    let factory = format!("0x{}", hex::encode(FACTORY));
    assert_eq!(
        b.submit_argv("sess-1", &GAME).unwrap(),
        s(&[
            "/usr/local/bin/basectl", "-c", "sepolia", "proofs", "submit", &game, "--private-key-file", "/etc/jg/key", "--session-id", "sess-1",
            "--zk-backend", "network", "--prover-rpc", "http://127.0.0.1:9000", "--factory", &factory, "--l1-rpc", "http://127.0.0.1:8545", "--yes",
            "--json",
        ])
    );
    // without a key file (send = false), basectl is never asked to submit
    assert!(backend(BackendKind::Network, DeliveryMode::BasectlSubmit, None).submit_argv("sess-1", &GAME).is_err());
    // the proof bytes command, its placeholders replaced
    assert_eq!(b.proof_bytes_argv("sess-1", &GAME), s(&["/opt/export-proof", "sess-1", &game]));
}

#[test]
fn basectl_l1_rpc_can_differ_from_ours() {
    let mut c = cfg(BackendKind::DryRun, DeliveryMode::ProofBytesCommand, "/b");
    c.basectl_l1_rpc = Some("http://10.0.0.5:8545".into());
    let b = BasectlBackend::new(c, FACTORY, "http://127.0.0.1:8545", None);
    let argv = b.propose_argv(&GAME, &PROVER, false);
    let i = argv.iter().position(|a| a == "--l1-rpc").unwrap();
    assert_eq!(argv[i + 1], "http://10.0.0.5:8545");
}

#[test]
fn the_propose_answer_names_the_session_of_our_game() {
    // `ProofsProposeJson` (camelCase), alloy addresses checksummed
    let out = r#"{
  "network": "sepolia",
  "proverRpc": "http://127.0.0.1:9000",
  "sessionId": "sepolia-network-6a6a-0001",
  "game": "0x6A6a6a6A6a6a6A6A6a6a6a6a6A6a6A6a6a6a6A6a",
  "proverAddress": "0xEeeEEEEEeeEEEEeeeeEEeeeeEEEEEEEEeeeeeEEe",
  "startBlock": 56903064,
  "endBlock": 56909063,
  "numBlocks": 6000,
  "intermediateRootInterval": 300,
  "status": "submitted"
}"#;
    assert_eq!(parse_propose(out, &GAME).unwrap(), "sepolia-network-6a6a-0001");
    assert!(parse_propose(out, &[0x11; 20]).is_err(), "the session of another game");
    assert!(parse_propose("{}", &GAME).is_err());
    assert!(parse_propose("not json", &GAME).is_err());
}

#[test]
fn the_status_answer_is_read_by_its_status_and_result() {
    let st = |status: &str, result: &str| format!(r#"{{"network":"sepolia","proverRpc":"http://127.0.0.1:9000","sessionId":"s","status":"{status}"{result}}}"#);
    assert_eq!(parse_status(&st("queued", ""), false).unwrap(), Poll::Pending);
    assert_eq!(parse_status(&st("running", ""), false).unwrap(), Poll::Pending);
    assert_eq!(parse_status(&st("succeeded", r#","result":{"proofType":"snark_plonk","zkVm":"sp1","proofBytes":869}"#), false).unwrap(), Poll::Ready);
    // a dry run: execution statistics, no proof bytes
    let dry = st(
        "succeeded",
        r#","result":{"proofType":"compressed","zkVm":"sp1","proofBytes":0,"executionStats":{"totalInstructionCycles":512000000000,"totalSp1Gas":1,"witnessGenerationMs":2,"executionMs":3}}"#,
    );
    assert_eq!(parse_status(&dry, true).unwrap(), Poll::DryRunDone { cycles: Some(512_000_000_000) });
    assert_eq!(parse_status(&st("succeeded", ""), true).unwrap(), Poll::DryRunDone { cycles: None });
    // a paid run that ends with anything but a PLONK proof is a failure, never something to send
    assert!(matches!(parse_status(&st("succeeded", r#","result":{"proofType":"compressed"}"#), false).unwrap(), Poll::Failed(_)));
    assert!(matches!(parse_status(&st("succeeded", ""), false).unwrap(), Poll::Failed(_)));
    assert_eq!(parse_status(&st("failed", r#","errorMessage":"witness generation failed""#), false).unwrap(), Poll::Failed("witness generation failed".into()));
    assert!(parse_status(&st("exploded", ""), false).is_err());
    assert!(parse_status("[]", false).is_err());
}

#[test]
fn the_submit_answer_is_a_verified_transaction_on_our_game() {
    let out = r#"{"network":"sepolia","l1Rpc":"http://127.0.0.1:8545","proverRpc":"http://127.0.0.1:9000","sessionId":"s",
        "game":"0x6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a6a","sender":"0xeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee","startBlock":1,"endBlock":6000,
        "numBlocks":6000,"proofBytes":869,"txHash":"0x00000000000000000000000000000000000000000000000000000000000abc01","blockNumber":12,"gasUsed":391000,"status":"verified"}"#;
    assert_eq!(parse_submit(out, &GAME).unwrap(), "0x00000000000000000000000000000000000000000000000000000000000abc01");
    assert!(parse_submit(&out.replace("verified", "pending"), &GAME).is_err());
    assert!(parse_submit(out, &[0x11; 20]).is_err());
    assert!(parse_submit(&out.replace("\"txHash\":\"0x00000000000000000000000000000000000000000000000000000000000abc01\",", ""), &GAME).is_err());
}

#[test]
fn proof_bytes_are_hex_of_a_zk_proof() {
    assert_eq!(parse_proof_bytes(" 0x015a093a2fabcd\n").unwrap(), vec![1, 0x5a, 0x09, 0x3a, 0x2f, 0xab, 0xcd]);
    assert_eq!(parse_proof_bytes("015a093a2fabcd").unwrap().len(), 7);
    assert!(parse_proof_bytes("0x005a093a2fabcd").is_err(), "a TEE proof is not ours to send");
    assert!(parse_proof_bytes("0x015a09").is_err(), "no room for the verifier selector");
    assert!(parse_proof_bytes("0xzz").is_err());
    assert!(parse_proof_bytes("").is_err());
}

#[test]
fn the_retry_delay_doubles_up_to_its_cap() {
    assert_eq!(backoff(1, 60, 3_600), 60);
    assert_eq!(backoff(2, 60, 3_600), 120);
    assert_eq!(backoff(3, 60, 3_600), 240);
    assert_eq!(backoff(7, 60, 3_600), 3_600);
    assert_eq!(backoff(200, 60, 3_600), 3_600);
    assert_eq!(backoff(0, 60, 3_600), 60);
}

#[test]
fn a_command_runs_bounded_its_output_captured() {
    let out = run_capture(&s(&["/bin/sh", "-c", "echo hello; echo oops >&2; exit 3"]), Duration::from_secs(10)).unwrap();
    assert_eq!(out.stdout.trim(), "hello");
    assert_eq!(out.stderr.trim(), "oops");
    assert_eq!(out.status.code(), Some(3));
    // killed with its process group past its time
    let start = Instant::now();
    let e = run_capture(&s(&["/bin/sh", "-c", "sleep 30 & sleep 30"]), Duration::from_secs(1)).unwrap_err();
    assert!(start.elapsed() < Duration::from_secs(10), "{:?}", start.elapsed());
    assert!(e.to_string().contains("timed out"), "{e}");
    assert!(run_capture(&s(&["/nonexistent/basectl"]), Duration::from_secs(1)).is_err());
    assert!(run_capture(&[], Duration::from_secs(1)).is_err());
}

#[test]
fn the_submitter_key_never_reaches_a_command_by_the_environment() {
    // the key is only ever the configured file: an operator's stray environment variable is not passed on
    std::env::set_var("BASECTL_SUBMITTER_PRIVATE_KEY", "0x01");
    let out = run_capture(&s(&["/bin/sh", "-c", "echo ${BASECTL_SUBMITTER_PRIVATE_KEY:-unset}"]), Duration::from_secs(10)).unwrap();
    std::env::remove_var("BASECTL_SUBMITTER_PRIVATE_KEY");
    assert_eq!(out.stdout.trim(), "unset");
}

/// A stand-in `basectl`: it logs its arguments and answers as basectl does.
fn fake_basectl(dir: &std::path::Path, status_answer: &str) -> std::path::PathBuf {
    let path = dir.join("basectl");
    let log = dir.join("calls.log");
    let script = format!(
        r#"#!/bin/sh
echo "$@" >> {log}
case "$4" in
  propose) printf '%s\n' '{{"network":"sepolia","proverRpc":"http://127.0.0.1:9000","sessionId":"sess-42","game":"'"$5"'","status":"submitted"}}' ;;
  status) printf '%s\n' '{status_answer}' ;;
  submit) printf '%s\n' '{{"game":"'"$5"'","txHash":"0x00000000000000000000000000000000000000000000000000000000000abc01","gasUsed":391000,"status":"verified"}}' ;;
  *) echo "unknown $4" >&2; exit 2 ;;
esac
"#,
        log = log.display()
    );
    std::fs::write(&path, script).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

#[test]
fn the_backend_drives_basectl_from_request_to_submission() {
    let dir = tempfile::tempdir().unwrap();
    let ready = r#"{"sessionId":"sess-42","status":"succeeded","result":{"proofType":"snark_plonk","proofBytes":869}}"#;
    let basectl = fake_basectl(dir.path(), ready);
    let key = dir.path().join("key");
    let mut b = BasectlBackend::new(cfg(BackendKind::Network, DeliveryMode::BasectlSubmit, basectl.to_str().unwrap()), FACTORY, "http://127.0.0.1:8545", Some(key.clone()));
    assert!(!b.dry_run());
    assert!(b.submits_itself());
    assert_eq!(b.start(&GAME, &PROVER, false).unwrap(), "sess-42");
    assert_eq!(b.poll("sess-42").unwrap(), Poll::Ready);
    let Delivery::Submitted { tx_hash } = b.deliver("sess-42", &GAME).unwrap() else { panic!("basectl submits itself") };
    assert_eq!(tx_hash, "0x00000000000000000000000000000000000000000000000000000000000abc01");
    let calls = std::fs::read_to_string(dir.path().join("calls.log")).unwrap();
    let lines: Vec<&str> = calls.lines().collect();
    assert_eq!(lines.len(), 3);
    assert!(lines[0].starts_with("-c sepolia proofs propose 0x6a6a"), "{}", lines[0]);
    assert!(lines[1].starts_with("-c sepolia proofs status sess-42"), "{}", lines[1]);
    assert!(lines[2].contains(&format!("--private-key-file {}", key.display())), "{}", lines[2]);
}

#[test]
fn a_dry_run_backend_never_submits() {
    let dir = tempfile::tempdir().unwrap();
    let done = r#"{"sessionId":"sess-42","status":"succeeded","result":{"proofType":"compressed","executionStats":{"totalInstructionCycles":77,"totalSp1Gas":1,"witnessGenerationMs":1,"executionMs":1}}}"#;
    let basectl = fake_basectl(dir.path(), done);
    let mut b = BasectlBackend::new(cfg(BackendKind::DryRun, DeliveryMode::BasectlSubmit, basectl.to_str().unwrap()), FACTORY, "http://127.0.0.1:8545", None);
    assert!(b.dry_run());
    assert_eq!(b.start(&GAME, &PROVER, false).unwrap(), "sess-42");
    assert_eq!(b.poll("sess-42").unwrap(), Poll::DryRunDone { cycles: Some(77) });
    assert!(b.deliver("sess-42", &GAME).is_err(), "a dry run has nothing to deliver");
    let calls = std::fs::read_to_string(dir.path().join("calls.log")).unwrap();
    assert!(!calls.contains("submit"), "{calls}");
}

#[test]
fn a_failing_basectl_is_an_error_with_its_words() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("basectl");
    std::fs::write(&path, "#!/bin/sh\necho 'Error: game is not provable: game already has a ZK proof' >&2\nexit 1\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut b = BasectlBackend::new(cfg(BackendKind::Network, DeliveryMode::ProofBytesCommand, path.to_str().unwrap()), FACTORY, "http://127.0.0.1:8545", Some("/k".into()));
    let e = b.start(&GAME, &PROVER, false).unwrap_err().to_string();
    assert!(e.contains("already has a ZK proof"), "{e}");
}

#[test]
fn the_proof_bytes_command_hands_the_bytes_and_nothing_is_submitted_by_it() {
    let dir = tempfile::tempdir().unwrap();
    let export = dir.path().join("export");
    std::fs::write(&export, "#!/bin/sh\n[ \"$1\" = sess-42 ] || exit 9\necho 0x015a093a2fabcd\n").unwrap();
    std::fs::set_permissions(&export, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut c = cfg(BackendKind::Network, DeliveryMode::ProofBytesCommand, "/unused");
    c.proof_bytes_command = vec![export.to_str().unwrap().into(), "{session}".into()];
    let mut b = BasectlBackend::new(c, FACTORY, "http://127.0.0.1:8545", Some("/k".into()));
    assert!(!b.submits_itself());
    assert_eq!(b.deliver("sess-42", &GAME).unwrap(), Delivery::Bytes(vec![1, 0x5a, 0x09, 0x3a, 0x2f, 0xab, 0xcd]));
}
