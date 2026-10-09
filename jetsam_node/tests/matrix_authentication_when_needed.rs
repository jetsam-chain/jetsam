// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A matrix the node waits for is not starved by the verifications that wait
//! for it.
//!
//! Testnet 3, 2026-10-09: a two-vCPU node restarted 29 blocks behind had to
//! authenticate its two held matrices again (203 MiB and 9.9 MiB) before it
//! could judge any v1.5 tip. Its snapshot candidate was judged again every
//! 10 s, refused each time for want of those matrices, and those judgements
//! (normal priority, both CPUs) left the authentication workers (SCHED_IDLE)
//! about 15 % of the machine: 630 s instead of the seed's 125 s.
//!
//! In its own process, pinned to two CPUs with the two shared workers of a
//! two-vCPU node, the shared pool kept busy the whole time: once a
//! verification has waited for the matrix, its authentication takes about
//! what two CPUs shared with that load give it, not what SCHED_IDLE does.

#![cfg(target_os = "linux")]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jetsam_chain::consensus::client_objects::{matrix_file_root, ClientObjectRules};
use jetsam_ivc_core::field_r1cs::{synthetic_satisfiable_bounded_dictionary, FieldR1cs};
use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
use jetsam_node::client_objects::ClientObjects;
use jetsam_p2p::client_object_protocol::MatrixFileId;
use jetsam_recursive::{HistoryStepClientForm, HistoryStepClientMatrixSet};
use rayon::prelude::*;

/// Each matrix of the client form is 2^MATRIX_LOG square: its authentication
/// lasts seconds on two workers in a debug build.
const MATRIX_LOG: usize = 15;

/// The public rules with this matrix in the client catalogue: only a listed
/// `D` is fetched or kept (review B3).
fn rules_listing(id: &MatrixFileId) -> ClientObjectRules {
    ClientObjectRules {
        catalogue: Box::leak(vec![id.matrix_digest].into_boxed_slice()),
        ..ClientObjectRules::CONSENSUS
    }
}

fn client_matrix() -> (HistoryStepClientForm, MatrixFileId, Vec<u8>) {
    let (matrix, _): (FieldR1cs, _) =
        synthetic_satisfiable_bounded_dictionary(MATRIX_LOG, MATRIX_LOG, 0x5EED, 64);
    let form = HistoryStepClientForm::new(
        FieldShape::of(&matrix),
        PcsParams {
            m: MATRIX_LOG + LOG_PACKING,
            log_inv_rate: 2,
            log_batch_size: 2,
            profile: Default::default(),
        },
        PublicIoSpec {
            io_slice: WitnessSlice {
                log2_len: 3,
                index: 1,
            },
            io_len: 8,
            claims: Vec::new(),
        },
        2,
    );
    let mut file = Vec::new();
    matrix.write_artifact(&mut file).unwrap();
    let id = MatrixFileId {
        matrix_digest: matrix.structural_statement_digest(),
        file_root: matrix_file_root(&file),
        file_len: file.len() as u32,
    };
    (form, id, file)
}

/// The client objects of a node restarted with `id` held on disk: opened,
/// the file not authenticated again yet.
fn restarted_with(
    directory: &std::path::Path,
    form: &HistoryStepClientForm,
    id: &MatrixFileId,
    file: &[u8],
) -> ClientObjects {
    let matrices = directory.join("client-objects").join("matrices");
    std::fs::create_dir_all(&matrices).unwrap();
    std::fs::write(
        matrices.join(format!(
            "{}.{}.{}.matrix",
            hex::encode(id.matrix_digest),
            hex::encode(id.file_root),
            id.file_len
        )),
        file,
    )
    .unwrap();
    let objects = ClientObjects::open(
        directory,
        form,
        Arc::new(HistoryStepClientMatrixSet::new(form)),
        &rules_listing(id),
    )
    .unwrap();
    assert!(objects.awaits_matrices());
    objects
}

/// This thread, and every thread it starts from now on, to the two least
/// busy CPUs it may run on (measured over a short spell).
fn pin_to_two_cpus() -> [usize; 2] {
    fn busy_and_total() -> Vec<(u64, u64)> {
        std::fs::read_to_string("/proc/stat")
            .unwrap()
            .lines()
            .filter(|line| line.starts_with("cpu") && !line.starts_with("cpu "))
            .map(|line| {
                let fields: Vec<u64> = line
                    .split_whitespace()
                    .skip(1)
                    .map(|field| field.parse().unwrap())
                    .collect();
                let total: u64 = fields.iter().sum();
                (total - fields[3] - fields[4], total)
            })
            .collect()
    }
    // SAFETY: a zeroed cpu_set_t is an empty set; the calls read and write
    // only the set passed to them.
    let mut allowed: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    assert_eq!(
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut allowed) },
        0
    );
    let before = busy_and_total();
    std::thread::sleep(Duration::from_millis(500));
    let after = busy_and_total();
    let mut cpus: Vec<(f64, usize)> = (0..before.len().min(after.len()))
        .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &allowed) })
        .map(|cpu| {
            let busy = after[cpu].0.saturating_sub(before[cpu].0) as f64;
            let total = after[cpu].1.saturating_sub(before[cpu].1).max(1) as f64;
            (busy / total, cpu)
        })
        .collect();
    assert!(cpus.len() >= 2, "this test needs two CPUs");
    cpus.sort_by(|a, b| a.0.total_cmp(&b.0));
    let chosen = [cpus[0].1, cpus[1].1];
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    unsafe {
        libc::CPU_SET(chosen[0], &mut set);
        libc::CPU_SET(chosen[1], &mut set);
    }
    assert_eq!(
        unsafe { libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) },
        0
    );
    chosen
}

/// The node judging its snapshot candidate again and again: parallel jobs
/// handed to the shared pool from outside it, back to back, until `stop`.
fn judge_again_and_again(stop: &AtomicBool) -> u64 {
    let mut judged = 0u64;
    while !stop.load(Ordering::SeqCst) {
        jetsam_miner::install_inbound_verifier_cpu(|| {
            (0..64u32)
                .into_par_iter()
                .map(|lane| {
                    jetsam_poseidon2b::native::poseidon2b_hash_byte_slices(
                        b"JTM/TEST/SNAPSHOT-CANDIDATE",
                        &[&lane.to_le_bytes(), &[0u8; 16384]],
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap();
        judged += 1;
    }
    judged
}

#[test]
fn a_matrix_a_verification_waits_for_is_not_starved_by_that_verification() {
    let cpus = pin_to_two_cpus();
    let plan = jetsam_miner::configure_process_cpu_budget_with_threads(
        jetsam_miner::ProcessCpuBudgetMode::ProofOnly,
        Some(2),
    )
    .unwrap();
    assert_eq!(plan.shared_pool_threads, 2);
    let (form, id, file) = client_matrix();

    // Alone: the two CPUs are the authentication's.
    let quiet = tempfile::tempdir().unwrap();
    let objects = restarted_with(quiet.path(), &form, &id, &file);
    let started = Instant::now();
    assert_eq!(objects.authenticate_held_files(), 1);
    let alone = started.elapsed();
    assert!(objects.holds_matrix(&id.matrix_digest));

    // Behind the network: the candidate is judged again and again, and the
    // first judgement answered "client matrix unavailable".
    let busy = tempfile::tempdir().unwrap();
    let objects = Arc::new(restarted_with(busy.path(), &form, &id, &file));
    let stop = Arc::new(AtomicBool::new(false));
    let judge = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || judge_again_and_again(&stop))
    };
    std::thread::sleep(Duration::from_millis(200));
    objects.note_matrices_needed();
    let (done, finished) = std::sync::mpsc::channel();
    {
        let objects = Arc::clone(&objects);
        std::thread::spawn(move || {
            let started = Instant::now();
            let held = objects.authenticate_held_files();
            let _ = done.send((held, started.elapsed()));
        });
    }
    // Four busy threads on two CPUs: about twice as long as alone. Starved
    // (SCHED_IDLE behind the judgements), many times longer: not waited for.
    let limit = alone * 4;
    let outcome = finished.recv_timeout(limit);
    stop.store(true, Ordering::SeqCst);
    let judged = judge.join().unwrap();
    let Ok((held, waited)) = outcome else {
        panic!(
            "CPUs {cpus:?}: the authentication the node waited for was not over after \
             {limit:?} (4x its {alone:?} alone) beside the {judged} judgements that waited \
             for it: it was starved"
        );
    };
    assert_eq!(held, 1);
    assert!(objects.holds_matrix(&id.matrix_digest));
    eprintln!(
        "CPUs {cpus:?}: authentication alone {alone:?}; while the candidate was judged \
         {judged} times: {waited:?} ({:.1}x)",
        waited.as_secs_f64() / alone.as_secs_f64()
    );
    assert!(
        judged >= 3,
        "the shared pool was not kept busy ({judged} judgements)"
    );
}
