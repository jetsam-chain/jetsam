// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The test catalogue's second form-Z entry (the computed batch of 97
//! statements of entry 1, `1070676e…`) written as an aggregator takes it:
//! `DIR/matrix.bin`, the registered matrix file, and `DIR/proof-N.bin`, a
//! form-Z proof of variant `N` (`bench_prover::zk_batch_client`).
//!
//! ```text
//! jetsam_client_zk_batch_demo --out DIR [--runs N] [--statements K]
//! ```

use std::time::Instant;

use bench_prover::zk_batch_client::{batch_of_97_form_z, BATCH_OF_97_CAPACITY};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let mut runs = 1u8;
    let mut count = BATCH_OF_97_CAPACITY;
    let mut out: Option<std::path::PathBuf> = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--runs" => runs = arguments.next().and_then(|v| v.parse().ok()).expect("--runs N (1..=255)"),
            "--statements" => {
                count = arguments
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|count| *count <= BATCH_OF_97_CAPACITY)
                    .expect("--statements K (0..=97)")
            }
            "--out" => out = Some(arguments.next().expect("--out DIR").into()),
            other => panic!("unknown argument {other:?}"),
        }
    }
    let out = out.expect("--out DIR");
    std::fs::create_dir_all(&out).expect("create --out");
    let variants: Vec<u8> = (0..runs).collect();
    let started = Instant::now();
    let files = batch_of_97_form_z(&variants, count).expect("the form-Z files of the batch of 97");
    std::fs::write(out.join("matrix.bin"), &files.matrix_file).expect("write matrix.bin");
    println!(
        "[zk-batch] matrix.bin: D {} file root {} length {} bytes",
        hex::encode(files.matrix_digest),
        hex::encode(jetsam_chain::consensus::client_objects::matrix_file_root(&files.matrix_file)),
        files.matrix_file.len()
    );
    for (run, proof) in files.proofs.iter().enumerate() {
        std::fs::write(out.join(format!("proof-{run}.bin")), proof).expect("write proof");
        println!("[zk-batch] proof-{run}.bin: {} bytes", proof.len());
    }
    println!("[zk-batch] {} proof(s) in {:.1} s", files.proofs.len(), started.elapsed().as_secs_f64());
}
