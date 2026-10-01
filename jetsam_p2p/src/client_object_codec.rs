// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Request-response codec for v1.5 client objects (M3 task 3.7): one client
//! proof bundle, one matrix manifest or one matrix chunk per request.
//!
//! The response echoes its request, so the requester's codec knows, before
//! it allocates, the exact length the payload must have, and refuses the
//! payload unless it is exactly the requested object: the bundle's byte
//! digest, the manifest's registered root, the chunk's digest. Payload memory
//! is admitted against the process-wide inbound budget shared with every
//! other large response.

use std::{io, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::{
    request_response::{self, ProtocolSupport},
    swarm::StreamProtocol,
};

use crate::{
    client_object_protocol::{
        matrix_chunk_matches, ClientProofId, Hash32, MatrixFileId, CLIENT_PROOF_BUNDLE_FIXED_BYTES,
        MAX_CLIENT_PROOF_BUNDLE_BYTES,
    },
    inbound_budget::process_global_inbound_budget,
    object_protocol::DataResponseStatus,
};
use jetsam_chain::consensus::client_objects::{ClientSubmission, CLIENT_MATRIX_MAX_FILE_BYTES};

const REQUEST_MAGIC: [u8; 4] = *b"JCQ1";
const RESPONSE_MAGIC: [u8; 4] = *b"JCS1";
const KIND_PROOF: u8 = 1;
const KIND_MANIFEST: u8 = 2;
const KIND_CHUNK: u8 = 3;
const NONE_LEN: u32 = u32::MAX;

/// Protocol path under the network's protocol id.
pub const CLIENT_OBJECTS_PROTOCOL_SUFFIX: &str = "/client/objects/1";

/// One client object, by its network identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClientObjectRequest {
    Proof(ClientProofId),
    MatrixManifest(MatrixFileId),
    /// Chunk `index` of a file; `digest` is the one its manifest lists.
    MatrixChunk {
        file: MatrixFileId,
        index: u32,
        digest: Hash32,
    },
}

impl ClientObjectRequest {
    /// The exact payload length this request admits, if well formed.
    pub fn expected_len(&self) -> Option<usize> {
        let file_ok =
            |file: &MatrixFileId| (1..=CLIENT_MATRIX_MAX_FILE_BYTES).contains(&file.file_len);
        match self {
            Self::Proof(id) => {
                let len = usize::try_from(id.encoded_len).ok()?;
                (CLIENT_PROOF_BUNDLE_FIXED_BYTES < len && len <= MAX_CLIENT_PROOF_BUNDLE_BYTES)
                    .then_some(len)
            }
            Self::MatrixManifest(file) => file_ok(file).then(|| file.manifest_len()),
            Self::MatrixChunk { file, index, .. } => {
                if file_ok(file) {
                    file.chunk_len(*index)
                } else {
                    None
                }
            }
        }
    }

    /// Whether `bytes` are exactly the requested object.
    pub fn matches(&self, bytes: &[u8]) -> bool {
        if self.expected_len() != Some(bytes.len()) {
            return false;
        }
        match self {
            Self::Proof(id) => id.matches_bytes(bytes),
            Self::MatrixManifest(file) => file.verify_manifest(bytes).is_some(),
            Self::MatrixChunk { index, digest, .. } => matrix_chunk_matches(*index, bytes, digest),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Proof(id) => {
                out.push(KIND_PROOF);
                out.extend_from_slice(&id.submission.matrix_digest);
                out.extend_from_slice(&id.submission.io_commitment);
                out.extend_from_slice(&id.bundle_digest);
                out.extend_from_slice(&id.encoded_len.to_le_bytes());
            }
            Self::MatrixManifest(file) => {
                out.push(KIND_MANIFEST);
                encode_file(file, out);
            }
            Self::MatrixChunk {
                file,
                index,
                digest,
            } => {
                out.push(KIND_CHUNK);
                encode_file(file, out);
                out.extend_from_slice(&index.to_le_bytes());
                out.extend_from_slice(digest);
            }
        }
    }

    /// Read one request body (after the magic).
    async fn read<T: AsyncRead + Unpin + Send>(io: &mut T) -> io::Result<Self> {
        let mut kind = [0u8; 1];
        io.read_exact(&mut kind).await?;
        let body_len = match kind[0] {
            KIND_PROOF => 100,
            KIND_MANIFEST => 68,
            KIND_CHUNK => 104,
            _ => return Err(invalid_data("unknown client object kind")),
        };
        let mut body = [0u8; 104];
        io.read_exact(&mut body[..body_len]).await?;
        let hash = |offset: usize| -> Hash32 { body[offset..offset + 32].try_into().unwrap() };
        let u32_at =
            |offset: usize| u32::from_le_bytes(body[offset..offset + 4].try_into().unwrap());
        let file = || MatrixFileId {
            matrix_digest: hash(0),
            file_root: hash(32),
            file_len: u32_at(64),
        };
        let request = match kind[0] {
            KIND_PROOF => Self::Proof(ClientProofId {
                submission: ClientSubmission {
                    matrix_digest: hash(0),
                    io_commitment: hash(32),
                },
                bundle_digest: hash(64),
                encoded_len: u32_at(96),
            }),
            KIND_MANIFEST => Self::MatrixManifest(file()),
            _ => Self::MatrixChunk {
                file: file(),
                index: u32_at(68),
                digest: hash(72),
            },
        };
        if request.expected_len().is_none() {
            return Err(invalid_data("client object request admits no object"));
        }
        Ok(request)
    }
}

/// The answer: the echoed request, a status and, when available, the object.
#[derive(Debug)]
pub struct ClientObjectResponse {
    pub request: ClientObjectRequest,
    pub status: DataResponseStatus,
    pub bytes: Option<Vec<u8>>,
    /// Inbound budget held until the node consumes the payload.
    pub inbound_memory_permit: Option<Arc<tokio::sync::OwnedSemaphorePermit>>,
}

impl PartialEq for ClientObjectResponse {
    fn eq(&self, other: &Self) -> bool {
        self.request == other.request && self.status == other.status && self.bytes == other.bytes
    }
}

impl Eq for ClientObjectResponse {}

impl ClientObjectResponse {
    pub fn ready(request: ClientObjectRequest, bytes: Vec<u8>) -> Self {
        Self {
            request,
            status: DataResponseStatus::Ready,
            bytes: Some(bytes),
            inbound_memory_permit: None,
        }
    }

    pub fn unavailable(request: ClientObjectRequest) -> Self {
        Self {
            request,
            status: DataResponseStatus::Ready,
            bytes: None,
            inbound_memory_permit: None,
        }
    }

    pub fn busy(request: ClientObjectRequest, retry_after_ms: u16) -> Self {
        Self {
            request,
            status: DataResponseStatus::Busy { retry_after_ms },
            bytes: None,
            inbound_memory_permit: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClientObjectCodec {
    inbound_budget: Arc<tokio::sync::Semaphore>,
}

impl Default for ClientObjectCodec {
    fn default() -> Self {
        Self {
            inbound_budget: process_global_inbound_budget(),
        }
    }
}

/// The request-response behaviour of the client-object protocol.
pub fn client_object_behaviour(
    protocol_id: &str,
) -> Result<request_response::Behaviour<ClientObjectCodec>, String> {
    let protocol =
        StreamProtocol::try_from_owned(format!("{protocol_id}{CLIENT_OBJECTS_PROTOCOL_SUFFIX}"))
            .map_err(|error| error.to_string())?;
    Ok(request_response::Behaviour::with_codec(
        ClientObjectCodec::default(),
        [(protocol, ProtocolSupport::Full)],
        request_response::Config::default()
            // A bundle is under 1 MB, a chunk 1 MiB: as the terminal protocol.
            .with_request_timeout(Duration::from_secs(60))
            .with_max_concurrent_streams(8),
    ))
}

#[async_trait]
impl request_response::Codec for ClientObjectCodec {
    type Protocol = StreamProtocol;
    type Request = ClientObjectRequest;
    type Response = ClientObjectResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_magic(io, REQUEST_MAGIC).await?;
        let request = ClientObjectRequest::read(io).await?;
        ensure_eof(io).await?;
        Ok(request)
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        read_magic(io, RESPONSE_MAGIC).await?;
        let request = ClientObjectRequest::read(io).await?;
        let expected = request
            .expected_len()
            .expect("a read request admits an object");
        let mut tail = [0u8; 8];
        io.read_exact(&mut tail).await?;
        if tail[1] != 0 {
            return Err(invalid_data("non-zero client object reserved byte"));
        }
        let retry_after_ms = u16::from_le_bytes(tail[2..4].try_into().unwrap());
        let status = match tail[0] {
            0 if retry_after_ms == 0 => DataResponseStatus::Ready,
            1 => DataResponseStatus::Busy { retry_after_ms },
            _ => return Err(invalid_data("invalid client object response status")),
        };
        if !status.is_canonical() {
            return Err(invalid_data("non-canonical client object response status"));
        }
        let declared = u32::from_le_bytes(tail[4..8].try_into().unwrap());
        let (bytes, inbound_memory_permit) = if declared == NONE_LEN {
            (None, None)
        } else {
            if matches!(status, DataResponseStatus::Busy { .. }) {
                return Err(invalid_data(
                    "busy client object response carries a payload",
                ));
            }
            // The exact length is known from the echoed request: refuse any
            // other before allocating anything.
            if declared as usize != expected {
                return Err(invalid_data(
                    "client object payload length is not the object's",
                ));
            }
            let permit = self.acquire_inbound(expected).await?;
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(expected)
                .map_err(|_| invalid_data("client object allocation failed"))?;
            bytes.resize(expected, 0);
            io.read_exact(&mut bytes).await?;
            // Hashing up to a megabyte is CPU work: keep it off the executor
            // that drives every other connection.
            let (exact, bytes) =
                tokio::task::spawn_blocking(move || (request.matches(&bytes), bytes))
                    .await
                    .map_err(io::Error::other)?;
            if !exact {
                return Err(invalid_data(
                    "client object payload is not the requested object",
                ));
            }
            (Some(bytes), permit)
        };
        ensure_eof(io).await?;
        Ok(ClientObjectResponse {
            request,
            status,
            bytes,
            inbound_memory_permit,
        })
    }

    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        request: Self::Request,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        if request.expected_len().is_none() {
            return Err(invalid_data("client object request admits no object"));
        }
        let mut frame = Vec::with_capacity(4 + 105);
        frame.extend_from_slice(&REQUEST_MAGIC);
        request.encode(&mut frame);
        io.write_all(&frame).await?;
        io.flush().await
    }

    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        response: Self::Response,
    ) -> io::Result<()>
    where
        T: AsyncWrite + Unpin + Send,
    {
        let ClientObjectResponse {
            request,
            status,
            bytes,
            inbound_memory_permit: _,
        } = response;
        if request.expected_len().is_none() {
            return Err(invalid_data("client object request admits no object"));
        }
        if !status.is_canonical() {
            return Err(invalid_data("non-canonical client object response status"));
        }
        if let Some(bytes) = bytes.as_deref() {
            if matches!(status, DataResponseStatus::Busy { .. }) {
                return Err(invalid_data(
                    "busy client object response carries a payload",
                ));
            }
            // The requester checks the object's digest; here, its exact
            // length only (a node serves what it verified on reception).
            if request.expected_len() != Some(bytes.len()) {
                return Err(invalid_data(
                    "refusing to serve a payload of another length",
                ));
            }
        }
        let mut frame = Vec::with_capacity(4 + 105 + 8);
        frame.extend_from_slice(&RESPONSE_MAGIC);
        request.encode(&mut frame);
        let mut tail = [0u8; 8];
        if let DataResponseStatus::Busy { retry_after_ms } = status {
            tail[0] = 1;
            tail[2..4].copy_from_slice(&retry_after_ms.to_le_bytes());
        }
        let declared = match bytes.as_deref() {
            Some(bytes) => bytes.len() as u32,
            None => NONE_LEN,
        };
        tail[4..8].copy_from_slice(&declared.to_le_bytes());
        frame.extend_from_slice(&tail);
        io.write_all(&frame).await?;
        if let Some(bytes) = bytes {
            io.write_all(&bytes).await?;
        }
        io.flush().await
    }
}

impl ClientObjectCodec {
    async fn acquire_inbound(
        &self,
        bytes: usize,
    ) -> io::Result<Option<Arc<tokio::sync::OwnedSemaphorePermit>>> {
        if bytes == 0 {
            return Ok(None);
        }
        let permits =
            u32::try_from(bytes).map_err(|_| invalid_data("client object byte budget overflow"))?;
        let permit = Arc::clone(&self.inbound_budget)
            .acquire_many_owned(permits)
            .await
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "client object byte budget closed",
                )
            })?;
        Ok(Some(Arc::new(permit)))
    }
}

fn encode_file(file: &MatrixFileId, out: &mut Vec<u8>) {
    out.extend_from_slice(&file.matrix_digest);
    out.extend_from_slice(&file.file_root);
    out.extend_from_slice(&file.file_len.to_le_bytes());
}

async fn read_magic<T: AsyncRead + Unpin>(io: &mut T, magic: [u8; 4]) -> io::Result<()> {
    let mut read = [0u8; 4];
    io.read_exact(&mut read).await?;
    if read != magic {
        return Err(invalid_data("invalid client object magic/version"));
    }
    Ok(())
}

async fn ensure_eof<T: AsyncRead + Unpin>(io: &mut T) -> io::Result<()> {
    let mut trailing = [0u8; 1];
    match io.read(&mut trailing).await? {
        0 => Ok(()),
        _ => Err(invalid_data("trailing bytes after client object frame")),
    }
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests;
