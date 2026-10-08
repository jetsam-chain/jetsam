// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

use super::*;
use crate::client_object_protocol::{ClientProofBundle, ClientProofId};
use futures::io::Cursor;
use jetsam_chain::consensus::client_objects::{
    matrix_file_root, ClientObject, CLIENT_MATRIX_CHUNK_BYTES,
};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{
    output_bitmap_bit, PagedSpendIntent, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};
use libp2p::request_response::Codec;

fn protocol() -> StreamProtocol {
    StreamProtocol::new("/jetsam-test/client/objects/1")
}

fn bundle(io: u8, proof_len: usize) -> ClientProofBundle {
    let submission = ClientSubmission {
        matrix_digest: [0x10; 32],
        io_commitment: [io; 32],
    };
    let mut inputs = [TxInput::dummy(); TX_INPUTS];
    inputs[0] = TxInput {
        slot_index: 100,
        amount: 2_000_000,
        creation_id: 1,
    };
    let mut outputs = [TxOutput::dummy(); TX_OUTPUTS];
    outputs[0] = TxOutput {
        slot_index: 101,
        amount: 0,
        owner: ClientObject::Submission(submission).marker(),
    };
    let payment = PagedSpendIntent::new(
        vec![TxPage {
            body: TxBody {
                epoch_anchor: [7u8; 32],
                fee: 2_000_000,
                input_owner: Address([0x51; 32]),
                inputs,
                outputs,
                validity_bitmap: 1
                    | output_bitmap_bit(0)
                    | PAGED_SPEND_START_BIT
                    | PAGED_SPEND_END_BIT,
                is_coinbase: false,
            },
        }],
        vec![0xA5; 64],
    )
    .unwrap();
    ClientProofBundle::new(submission, payment, vec![io; proof_len]).unwrap()
}

fn matrix_file() -> (Vec<u8>, MatrixFileId) {
    let file: Vec<u8> = (0..CLIENT_MATRIX_CHUNK_BYTES + 777)
        .map(|index| (index % 253) as u8)
        .collect();
    let id = MatrixFileId {
        matrix_digest: [0xD1; 32],
        file_root: matrix_file_root(&file),
        file_len: file.len() as u32,
    };
    (file, id)
}

async fn request_round_trip(request: ClientObjectRequest) -> io::Result<ClientObjectRequest> {
    let mut codec = ClientObjectCodec::default();
    let mut wire = Cursor::new(Vec::new());
    codec.write_request(&protocol(), &mut wire, request).await?;
    let mut read = Cursor::new(wire.into_inner());
    codec.read_request(&protocol(), &mut read).await
}

async fn encode_response(response: ClientObjectResponse) -> io::Result<Vec<u8>> {
    let mut codec = ClientObjectCodec::default();
    let mut wire = Cursor::new(Vec::new());
    codec
        .write_response(&protocol(), &mut wire, response)
        .await?;
    Ok(wire.into_inner())
}

async fn decode_response(bytes: Vec<u8>) -> io::Result<ClientObjectResponse> {
    let mut codec = ClientObjectCodec::default();
    codec
        .read_response(&protocol(), &mut Cursor::new(bytes))
        .await
}

#[tokio::test]
async fn requests_round_trip_and_reject_malformed_frames() {
    let (file, matrix) = matrix_file();
    let digests = matrix
        .verify_manifest(&MatrixFileId::manifest_of(&file))
        .unwrap();
    let requests = [
        ClientObjectRequest::Proof(bundle(1, 100).id()),
        ClientObjectRequest::MatrixManifest(matrix),
        ClientObjectRequest::MatrixChunk {
            file: matrix,
            index: 1,
            digest: digests[1],
        },
    ];
    for request in requests {
        assert_eq!(request_round_trip(request).await.unwrap(), request);
        let mut codec = ClientObjectCodec::default();
        let mut wire = Cursor::new(Vec::new());
        codec
            .write_request(&protocol(), &mut wire, request)
            .await
            .unwrap();
        let bytes = wire.into_inner();
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(codec
            .read_request(&protocol(), &mut Cursor::new(trailing))
            .await
            .is_err());
        assert!(codec
            .read_request(
                &protocol(),
                &mut Cursor::new(bytes[..bytes.len() - 1].to_vec())
            )
            .await
            .is_err());
        let mut magic = bytes.clone();
        magic[0] ^= 1;
        assert!(codec
            .read_request(&protocol(), &mut Cursor::new(magic))
            .await
            .is_err());
        let mut kind = bytes.clone();
        kind[4] = 9;
        assert!(codec
            .read_request(&protocol(), &mut Cursor::new(kind))
            .await
            .is_err());
    }
    // A request no object can answer is refused before it is sent.
    let mut too_long = bundle(1, 100).id();
    too_long.encoded_len = (MAX_CLIENT_PROOF_BUNDLE_BYTES + 1) as u32;
    assert!(request_round_trip(ClientObjectRequest::Proof(too_long))
        .await
        .is_err());
    let beyond = ClientObjectRequest::MatrixChunk {
        file: matrix,
        index: 2,
        digest: [0; 32],
    };
    assert!(request_round_trip(beyond).await.is_err());
}

#[tokio::test]
async fn exact_objects_round_trip_and_altered_ones_are_refused() {
    let bundle = bundle(1, 409_000);
    let bytes = bundle.encode();
    let request = ClientObjectRequest::Proof(bundle.id());
    let wire = encode_response(ClientObjectResponse::ready(request, bytes.clone()))
        .await
        .unwrap();
    let response = decode_response(wire.clone()).await.unwrap();
    assert_eq!(
        response,
        ClientObjectResponse::ready(request, bytes.clone())
    );
    assert!(response.inbound_memory_permit.is_some());

    // One altered payload byte: refused by the requester's codec.
    let mut altered = wire.clone();
    let last = altered.len() - 1;
    altered[last] ^= 1;
    assert_eq!(
        decode_response(altered).await.unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );
    // Truncated: refused.
    assert!(decode_response(wire[..wire.len() - 1].to_vec())
        .await
        .is_err());
    // Trailing: refused.
    let mut trailing = wire.clone();
    trailing.push(0);
    assert!(decode_response(trailing).await.is_err());

    // A server never sends a payload of another length; a same-length
    // altered payload is the requester's to refuse (above).
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(
        encode_response(ClientObjectResponse::ready(request, longer))
            .await
            .is_err()
    );
    let mut wrong = bytes.clone();
    wrong[10] ^= 1;
    let lying = encode_response(ClientObjectResponse::ready(request, wrong))
        .await
        .unwrap();
    assert_eq!(
        decode_response(lying).await.unwrap_err().kind(),
        io::ErrorKind::InvalidData
    );

    // Unavailable and busy carry nothing.
    for response in [
        ClientObjectResponse::unavailable(request),
        ClientObjectResponse::busy(request, 500),
    ] {
        let expected = ClientObjectResponse {
            request: response.request,
            status: response.status,
            bytes: None,
            inbound_memory_permit: None,
        };
        let wire = encode_response(response).await.unwrap();
        assert_eq!(decode_response(wire).await.unwrap(), expected);
    }
}

#[tokio::test]
async fn a_declared_length_other_than_the_objects_is_refused_before_allocation() {
    let bundle = bundle(1, 1_000);
    let bytes = bundle.encode();
    let request = ClientObjectRequest::Proof(bundle.id());
    let wire = encode_response(ClientObjectResponse::ready(request, bytes.clone()))
        .await
        .unwrap();
    // The length field sits just before the payload.
    let length_at = wire.len() - bytes.len() - 4;
    for declared in [bytes.len() as u32 + 1, bytes.len() as u32 - 1, u32::MAX - 1] {
        let mut lying = wire[..length_at].to_vec();
        lying.extend_from_slice(&declared.to_le_bytes());
        // An empty budget: any allocation attempt would wait forever.
        let mut codec = ClientObjectCodec {
            inbound_budget: Arc::new(tokio::sync::Semaphore::new(0)),
        };
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            codec.read_response(&protocol(), &mut Cursor::new(lying)),
        )
        .await
        .expect("refused before waiting for budget");
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::InvalidData);
    }
}

#[tokio::test]
async fn matrix_manifests_and_chunks_are_checked_on_reception() {
    let (file, matrix) = matrix_file();
    let manifest = MatrixFileId::manifest_of(&file);
    let digests = matrix.verify_manifest(&manifest).unwrap();
    let manifest_request = ClientObjectRequest::MatrixManifest(matrix);
    let wire = encode_response(ClientObjectResponse::ready(
        manifest_request,
        manifest.clone(),
    ))
    .await
    .unwrap();
    assert_eq!(
        decode_response(wire.clone()).await.unwrap().bytes,
        Some(manifest.clone())
    );
    let mut altered = wire;
    let last = altered.len() - 1;
    altered[last] ^= 1;
    assert!(decode_response(altered).await.is_err());

    for (index, chunk) in file.chunks(CLIENT_MATRIX_CHUNK_BYTES).enumerate() {
        let request = ClientObjectRequest::MatrixChunk {
            file: matrix,
            index: index as u32,
            digest: digests[index],
        };
        let wire = encode_response(ClientObjectResponse::ready(request, chunk.to_vec()))
            .await
            .unwrap();
        assert_eq!(
            decode_response(wire.clone())
                .await
                .unwrap()
                .bytes
                .as_deref(),
            Some(chunk)
        );
        let mut altered = wire;
        let last = altered.len() - 1;
        altered[last] ^= 1;
        assert!(decode_response(altered).await.is_err());
    }
    // The other chunk's bytes under this index: refused at both ends.
    let swapped = ClientObjectRequest::MatrixChunk {
        file: matrix,
        index: 1,
        digest: digests[1],
    };
    assert!(encode_response(ClientObjectResponse::ready(
        swapped,
        file[..CLIENT_MATRIX_CHUNK_BYTES].to_vec()
    ))
    .await
    .is_err());
    let _ = ClientProofId::of_bytes;
}

/// Decision of 2026-10-08 (v1.5.0): a registered matrix file may weigh up to
/// 256 MiB. A computed batch's file (capacity 128: 212 931 651 bytes, 204
/// chunks) travels like any other: its manifest and its chunks, the last one
/// short, are admitted, served and checked on reception. A file one byte
/// over 256 MiB admits no request: a node neither asks for it nor answers.
#[tokio::test]
async fn a_batch_matrix_file_travels_in_chunks_and_one_over_the_cap_admits_none() {
    use crate::client_object_protocol::MAX_CLIENT_MATRIX_MANIFEST_BYTES;
    use jetsam_chain::consensus::client_objects::{
        matrix_file_chunk_digest, matrix_file_root_from_chunk_digests,
    };
    const MIB: usize = 1024 * 1024;
    let file_len = 212_931_651usize;
    let count = file_len.div_ceil(CLIENT_MATRIX_CHUNK_BYTES);
    assert_eq!(count, 204);
    // The first and the last chunk are real bytes; the manifest lists their
    // digests and stand-ins for the others (the transport checks a manifest
    // against the registered root, and a chunk against its manifest digest).
    let first: Vec<u8> = (0..CLIENT_MATRIX_CHUNK_BYTES)
        .map(|index| (index % 251) as u8)
        .collect();
    let last: Vec<u8> = (0..file_len - (count - 1) * CLIENT_MATRIX_CHUNK_BYTES)
        .map(|index| (index % 241) as u8)
        .collect();
    let mut digests: Vec<Hash32> = (0..count).map(|index| [index as u8; 32]).collect();
    digests[0] = matrix_file_chunk_digest(0, &first);
    digests[count - 1] = matrix_file_chunk_digest((count - 1) as u32, &last);
    let file = MatrixFileId {
        matrix_digest: [0xB1; 32],
        file_root: matrix_file_root_from_chunk_digests(file_len as u64, &digests),
        file_len: file_len as u32,
    };
    let manifest = digests.concat();
    let request = ClientObjectRequest::MatrixManifest(file);
    assert_eq!(request.expected_len(), Some(32 * count));
    assert_eq!(request_round_trip(request).await.unwrap(), request);
    let wire = encode_response(ClientObjectResponse::ready(request, manifest.clone()))
        .await
        .unwrap();
    assert_eq!(decode_response(wire).await.unwrap().bytes, Some(manifest));
    for (index, chunk) in [(0, &first), (count - 1, &last)] {
        let request = ClientObjectRequest::MatrixChunk {
            file,
            index: index as u32,
            digest: digests[index],
        };
        assert_eq!(request.expected_len(), Some(chunk.len()));
        assert_eq!(request_round_trip(request).await.unwrap(), request);
        let wire = encode_response(ClientObjectResponse::ready(request, chunk.clone()))
            .await
            .unwrap();
        assert_eq!(
            decode_response(wire).await.unwrap().bytes.as_deref(),
            Some(chunk.as_slice())
        );
    }
    for (len, admitted) in [(256 * MIB, true), (256 * MIB + 1, false)] {
        let file = MatrixFileId {
            file_len: len as u32,
            ..file
        };
        let manifest = ClientObjectRequest::MatrixManifest(file);
        let chunk = ClientObjectRequest::MatrixChunk {
            file,
            index: 0,
            digest: [0; 32],
        };
        assert_eq!(manifest.expected_len().is_some(), admitted, "{len} bytes");
        assert_eq!(chunk.expected_len().is_some(), admitted, "{len} bytes");
        assert_eq!(
            request_round_trip(manifest).await.is_ok(),
            admitted,
            "{len} bytes"
        );
    }
    assert_eq!(MAX_CLIENT_MATRIX_MANIFEST_BYTES, 32 * 256);
}
