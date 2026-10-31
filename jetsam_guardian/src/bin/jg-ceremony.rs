// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-ceremony` — the dealing (design §6.2), on an OFFLINE machine, in RAM.
//!
//! * `run --machine-keys <file> --out <dir> --dice-fd <n> --network <hex> --l1-chain-id <n> --l1-bridge <hex> [--date <d>]`:
//!   64 bytes of the OS generator mixed with ≥ 100 dice throws read from the
//!   descriptor; Shamir 5 of 7, Feldman, the internal checks, the test
//!   signature, the revocation certificate (one file, to be sealed and kept
//!   OFFLINE by the user, D34), each share sealed for its machine key;
//! * `test --label <label> --machine-keys <file> --out <dir> …`: the same with
//!   the TEST coefficients of `label` (`fixture` = the rehearsal key pinned in
//!   the node's rehearsal profile). TEST ONLY.
//!
//! `--machine-keys`: 7 lines, the machines' X25519 public keys (hex), in the
//! order of the indices 1..=7.

use std::io::Read;
use std::os::unix::io::FromRawFd;
use std::path::PathBuf;

use jetsam_guardian::ceremony::{check_dealing, coefficients, ikm_from_entropy, revocation_certificate, test_message, write_outputs};
use jetsam_guardian::threshold::{deal, test_coefficients};
use jetsam_guardian::Domain;

fn run() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let get = |n: &str| a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned().ok_or_else(|| anyhow::anyhow!("missing {n}"));
    let mode = a.get(1).cloned().unwrap_or_default();
    let keys: Vec<[u8; 32]> = std::fs::read_to_string(get("--machine-keys")?)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(jetsam_guardian::parse32)
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(keys.len() == 7, "7 machine keys, one per index");
    let out = PathBuf::from(get("--out")?);
    let bridge: [u8; 20] = hex::decode(get("--l1-bridge")?.trim_start_matches("0x"))?.try_into().map_err(|_| anyhow::anyhow!("bridge"))?;
    let domain = Domain { network: jetsam_guardian::parse32(&get("--network")?)?, l1_chain_id: get("--l1-chain-id")?.parse()?, l1_bridge: bridge };
    let coeffs = match mode.as_str() {
        "test" => test_coefficients(&get("--label")?),
        "run" => {
            let mut os = [0u8; 64];
            getrandom::getrandom(&mut os).map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut dice = String::new();
            let fd: i32 = get("--dice-fd")?.parse()?;
            unsafe { std::fs::File::from_raw_fd(fd) }.read_to_string(&mut dice)?;
            coefficients(&*ikm_from_entropy(&os, &dice)?)?
        }
        _ => anyhow::bail!("jg-ceremony run | test"),
    };
    let dealing = deal(&coeffs).map_err(|e| anyhow::anyhow!("{e}"))?;
    let date = get("--date").unwrap_or_else(|_| jetsam_guardian::feed::utc_day(jetsam_guardian::now_unix()));
    let m = test_message(&dealing.group_key, &date);
    let test_sig = check_dealing(&dealing, &coeffs[0], &m)?;
    let cert = revocation_certificate(&coeffs[0], &domain)?;
    let public = write_outputs(&out, &dealing, &keys, 1)?;
    std::fs::write(out.join("revocation-certificate.json"), serde_json::json!({"body": hex::encode(&cert.body), "hints": hex::encode(&cert.hints), "_note": "ONE copy, offline, sealed (D34)"}).to_string())?;
    println!("{}", serde_json::json!({"group_key": public["group_key"], "commitments": public["commitments"], "test_message": hex::encode(m),
        "test_signature": hex::encode(test_sig), "date": date, "mode": mode}));
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("jg-ceremony: {e:#}");
        std::process::exit(1);
    }
}
