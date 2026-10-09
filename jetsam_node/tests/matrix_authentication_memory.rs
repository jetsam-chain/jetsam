// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! What authenticating a matrix file holds in memory.
//!
//! A decoded client matrix weighs about three times its file (0.62 GiB for
//! the 203 MiB file of a batch of 128 statements), and every node
//! authenticates a registered matrix at once, and again at each restart.
//! Testnet 3, 2026-10-09: on the seed the resident memory rose by 1.05 GiB
//! at the authentication for 0.62 GiB kept, the received chunks copied into
//! one file, the file held beside its decoded matrix for the whole
//! structural digest.
//!
//! In its own process: the allocator counts every live byte of it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use jetsam_chain::consensus::client_objects::{
    matrix_file_root, ClientObjectRules, CLIENT_MATRIX_CHUNK_BYTES, CLIENT_MATRIX_MAX_FILE_BYTES,
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

/// The live bytes of this process, and the most it held since the last reset.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let live = LIVE.fetch_add(by, Ordering::SeqCst) + by;
    PEAK.fetch_max(live, Ordering::SeqCst);
}

// SAFETY: every call is forwarded to the system allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc(layout);
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = System.alloc_zeroed(layout);
        if !pointer.is_null() {
            grew(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        System.dealloc(pointer, layout);
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = System.realloc(pointer, layout, new_size);
        if !moved.is_null() {
            grew(new_size);
            LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Each matrix of the client form is 2^MATRIX_LOG square: its file is a few
/// MiB, far above what a decode or a digest allocates on the side.
const MATRIX_LOG: usize = 16;

/// A client form of `MATRIX_LOG` rows, a client matrix of it, its file and
/// its registered identity.
fn client_matrix() -> (HistoryStepClientForm, MatrixFileId, Vec<u8>) {
    let (matrix, _): (FieldR1cs, _) =
        synthetic_satisfiable_bounded_dictionary(MATRIX_LOG, MATRIX_LOG, 0x3E3, 64);
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

/// A matrix file is never held whole beside its decoded matrix.
///
/// - Fetched: the received chunks are the file. Authenticating it holds,
///   beyond them, less than its decoded matrix: its bytes are on disk, and
///   freed, before it is decoded, never copied whole.
/// - Held across a restart: authenticating it again holds less than its
///   decoded matrix and half its file: its chunks are read and hashed a few
///   at a time, then the matrix is decoded from disk.
///
/// One test: the counts are the whole process's.
#[test]
fn a_matrix_file_is_never_held_beside_its_decoded_matrix() {
    // One authentication worker: after the restart the file's chunks are
    // read and hashed one at a time.
    jetsam_miner::configure_process_cpu_budget_with_threads(
        jetsam_miner::ProcessCpuBudgetMode::ProofOnly,
        Some(1),
    )
    .unwrap();
    let (form, id, file) = client_matrix();
    let file_len = file.len();
    assert!(
        file_len > CLIENT_MATRIX_CHUNK_BYTES,
        "a file of several chunks"
    );
    let before = LIVE.load(Ordering::SeqCst);
    let decoded = FieldR1cs::read_artifact_unbound(
        &mut &file[..],
        form.shape(),
        CLIENT_MATRIX_MAX_FILE_BYTES as usize,
    )
    .unwrap();
    let matrix_bytes = LIVE.load(Ordering::SeqCst) - before;
    drop(decoded);

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
    let assembled = fetched(&objects, id, &file);
    drop(file);
    let received = LIVE.load(Ordering::SeqCst);
    PEAK.store(received, Ordering::SeqCst);
    assert_eq!(objects.authenticate_assembled(assembled).unwrap(), id);
    let above = PEAK.load(Ordering::SeqCst) - received;
    assert!(objects.holds_matrix(&id.matrix_digest));
    eprintln!(
        "file {file_len} B, decoded matrix {matrix_bytes} B; held above the received chunks \
         during the authentication: {above} B"
    );
    assert!(
        matrix_bytes > file_len * 2,
        "a decoded matrix ({matrix_bytes} B) is expected to outweigh its file ({file_len} B)"
    );
    assert!(
        above < matrix_bytes,
        "the authentication held {above} B above the received chunks: the file ({file_len} B) \
         beside its decoded matrix ({matrix_bytes} B)"
    );
    // Nothing staged is left behind: only the held file, under its name.
    let names: Vec<String> = std::fs::read_dir(directory.path().join("client-objects/matrices"))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
    assert!(names[0].ends_with(".matrix"), "{names:?}");

    // A restart: the file held on disk is authenticated again.
    drop(objects);
    let objects = ClientObjects::open(
        directory.path(),
        &form,
        Arc::new(HistoryStepClientMatrixSet::new(&form)),
        &ClientObjectRules::CONSENSUS,
    )
    .unwrap();
    let reopened = LIVE.load(Ordering::SeqCst);
    PEAK.store(reopened, Ordering::SeqCst);
    assert_eq!(objects.authenticate_held_files(), 1);
    let above = PEAK.load(Ordering::SeqCst) - reopened;
    assert!(objects.holds_matrix(&id.matrix_digest));
    eprintln!("held above the reopened objects during the restart's authentication: {above} B");
    assert!(
        above < matrix_bytes + file_len / 2,
        "the restart's authentication held {above} B: the file ({file_len} B) beside its \
         decoded matrix ({matrix_bytes} B)"
    );
}
