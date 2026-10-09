// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! A block is never verified behind a client matrix's authentication.
//!
//! Testnet 3, 2026-10-09: when a batch of 128 statements was registered, its
//! 203 MiB matrix was authenticated on every node at once, on the shared CPU
//! pool where blocks are verified. Rayon runs a job handed to a pool from
//! outside only once its workers find nothing else to do, and a worker that
//! waits inside one job may run another on top of it, so on the two-vCPU seed
//! the verification of block 1222 ended with the authentication (95 s) and
//! took 102.6 s instead of 9-11 s.
//!
//! In its own process: the shared pool is configured once per process, here
//! with the two workers of a two-vCPU node. That the authentication workers
//! run at the lowest CPU priority is checked with the client objects' unit
//! tests: on a shared host, how long a verification takes on two CPUs
//! depends on what else runs there.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use jetsam_chain::consensus::client_objects::{
    matrix_file_root, ClientObjectRules, CLIENT_MATRIX_CHUNK_BYTES,
};
use jetsam_ivc_core::field_r1cs::{synthetic_satisfiable_bounded_dictionary, FieldR1cs};
use jetsam_ivc_core::pcs::{PcsParams, LOG_PACKING};
use jetsam_ivc_core::proof::FieldShape;
use jetsam_ivc_core::public_io::{PublicIoSpec, WitnessSlice};
use jetsam_node::client_objects::{AssembledMatrixFile, ClientObjectFetched, ClientObjects};
use jetsam_p2p::client_object_codec::ClientObjectRequest;
use jetsam_p2p::client_object_protocol::MatrixFileId;
use jetsam_recursive::{HistoryStepClientForm, HistoryStepClientMatrixSet};
use libp2p::PeerId;
use rayon::prelude::*;

/// Each matrix of the client form is 2^MATRIX_LOG square: its authentication
/// lasts seconds on two workers in a debug build.
const MATRIX_LOG: usize = 15;

/// A client form of `MATRIX_LOG` rows, a client matrix of it, its file and
/// its registered identity.
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

/// `id` fetched from a peer serving `file`, every chunk received.
fn fetched(objects: &ClientObjects, id: MatrixFileId, file: &[u8]) -> AssembledMatrixFile {
    objects.want_matrices([id]);
    let peers = [PeerId::random()];
    loop {
        let fetches = objects.next_requests(&peers, Instant::now());
        assert!(!fetches.is_empty(), "the fetch stalled");
        for fetch in fetches {
            let bytes = match fetch.request {
                ClientObjectRequest::MatrixManifest(_) => MatrixFileId::manifest_of(file),
                ClientObjectRequest::MatrixChunk { index, .. } => file
                    .chunks(CLIENT_MATRIX_CHUNK_BYTES)
                    .nth(index as usize)
                    .unwrap()
                    .to_vec(),
                ClientObjectRequest::Proof(_) => unreachable!(),
            };
            match objects.on_fetched(fetch.token, &bytes).unwrap() {
                ClientObjectFetched::MatrixAssembled(assembled) => return assembled,
                ClientObjectFetched::Nothing => {}
                other => panic!("{other:?}"),
            }
        }
    }
}

/// A stand-in for a block's verification: a parallel job handed to the
/// shared pool from outside it, as the node verifies a block
/// (`install_inbound_verifier_cpu`). Returns how long it took.
fn verify_a_block() -> Duration {
    let started = Instant::now();
    jetsam_miner::install_inbound_verifier_cpu(|| {
        (0..32u32)
            .into_par_iter()
            .map(|lane| {
                jetsam_poseidon2b::native::poseidon2b_hash_byte_slices(
                    b"JTM/TEST/BLOCK-VERIFICATION",
                    &[&lane.to_le_bytes(), &[0u8; 2048]],
                )
            })
            .collect::<Vec<_>>()
    })
    .unwrap();
    started.elapsed()
}

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort();
    durations[durations.len() / 2]
}

/// Blocks verified during a matrix's authentication, run as the node runs
/// it (from a blocking thread, `spawn_matrix_authentication`): none waits for
/// it. The node is not verifying all the time: each verification is followed
/// by an idle spell as long.
#[test]
fn a_block_verification_does_not_wait_for_a_matrix_authentication() {
    let plan = jetsam_miner::configure_process_cpu_budget_with_threads(
        jetsam_miner::ProcessCpuBudgetMode::ProofOnly,
        Some(2),
    )
    .unwrap();
    assert_eq!(plan.shared_pool_threads, 2);
    let (form, id, file) = client_matrix();
    let directory = tempfile::tempdir().unwrap();
    let objects = Arc::new(
        ClientObjects::open(
            directory.path(),
            &form,
            Arc::new(HistoryStepClientMatrixSet::new(&form)),
            &ClientObjectRules::CONSENSUS,
        )
        .unwrap(),
    );
    let alone = median((0..16).map(|_| verify_a_block()).collect());

    let assembled = fetched(&objects, id, &file);
    let authenticating = Arc::new(AtomicBool::new(true));
    let authentication = {
        let objects = Arc::clone(&objects);
        let authenticating = Arc::clone(&authenticating);
        std::thread::spawn(move || {
            let started = Instant::now();
            let outcome = objects.authenticate_assembled(assembled);
            authenticating.store(false, Ordering::SeqCst);
            (
                outcome.map_err(|error| error.to_string()),
                started.elapsed(),
            )
        })
    };
    // Every verification started while the matrix is authenticated counts,
    // however late it ends.
    let mut during = Vec::new();
    while authenticating.load(Ordering::SeqCst) {
        during.push(verify_a_block());
        std::thread::sleep(alone);
    }
    let (outcome, authenticated_in) = authentication.join().unwrap();
    assert_eq!(outcome, Ok(id));
    assert!(objects.holds_matrix(&id.matrix_digest));
    let (verified, worst) = (during.len(), during.iter().max().copied().unwrap());
    eprintln!(
        "verification alone (median of 16) {alone:?}; authentication {authenticated_in:?}; \
         {verified} verifications during it: median {:?}, slowest {worst:?}",
        median(during)
    );
    assert!(
        worst < authenticated_in / 5,
        "a block verification took {worst:?} during a matrix authentication of \
         {authenticated_in:?} (alone: {alone:?}): it waited for the authentication"
    );
    assert!(
        authenticated_in > alone * 20 && verified >= 10,
        "the authentication ({authenticated_in:?}, {verified} verifications) is too short \
         beside a verification ({alone:?}) for this test to tell anything"
    );
}
