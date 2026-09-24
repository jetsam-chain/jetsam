// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Exact network-v7 profile advertised before a connection becomes usable.

use std::io;

use async_trait::async_trait;
use futures::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use libp2p::{request_response, swarm::StreamProtocol};
use jetsam_chain::{
    block_header::block_id,
    consensus::wire_limits::{MAX_BLOCK_BYTES, MAX_HISTORY_STEP_TERMINAL_BYTES, MAX_SEGMENT_BYTES},
    consensus::{genesis_header, params::CONSENSUS_FINALITY_DEPTH},
    history_step::{HISTORY_STEP_CLASS_COUNT, HISTORY_STEP_TERMINAL_VERSION},
};
use jetsam_poseidon2b::native::poseidon2b_hash_bytes;

use crate::header_sync_codec::MAX_HEADERS_PER_BATCH;

const PROFILE_ID_DOMAIN: &[u8] = b"JTM/P2P/NETWORK-PROFILE/V7";
// JETSAM: the leading byte E replaces the upstream magic's initial, so no
// stream or file magic is byte-identical to Parano1d's.
const REQUEST_MAGIC: [u8; 4] = *b"JPQ6";
const RESPONSE_MAGIC: [u8; 4] = *b"JPS6";
const REQUEST_BYTES: usize = 4 + 32;
const PROFILE_BYTES: usize = 2 + 32 + 32 + 4 + 4 + 4 + 2 + 2 + 1 + 1;
const RESPONSE_BYTES: usize = 4 + PROFILE_BYTES + 32;

pub const NETWORK_WIRE_VERSION: u16 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkProfile {
    pub wire_version: u16,
    pub genesis_hash: [u8; 32],
    /// Semantic identity of the preflight-authenticated HistoryStep class bank.
    /// It is independent of a Git commit, branch, host, and build timestamp.
    pub history_proof_bank_id: [u8; 32],
    pub max_block_bytes: u32,
    pub max_terminal_bytes: u32,
    pub max_segment_bytes: u32,
    pub max_header_batch: u16,
    pub finality_depth: u16,
    pub history_terminal_version: u8,
    pub history_class_count: u8,
    pub profile_id: [u8; 32],
}

impl NetworkProfile {
    pub fn for_proof_bank(history_proof_bank_id: [u8; 32]) -> Self {
        let mut profile = Self {
            wire_version: NETWORK_WIRE_VERSION,
            genesis_hash: block_id(&genesis_header()),
            history_proof_bank_id,
            max_block_bytes: u32::try_from(MAX_BLOCK_BYTES).expect("block cap fits u32"),
            max_terminal_bytes: u32::try_from(MAX_HISTORY_STEP_TERMINAL_BYTES)
                .expect("terminal cap fits u32"),
            max_segment_bytes: u32::try_from(MAX_SEGMENT_BYTES).expect("segment cap fits u32"),
            max_header_batch: u16::try_from(MAX_HEADERS_PER_BATCH)
                .expect("header batch cap fits u16"),
            finality_depth: u16::try_from(CONSENSUS_FINALITY_DEPTH)
                .expect("finality depth fits u16"),
            history_terminal_version: HISTORY_STEP_TERMINAL_VERSION,
            history_class_count: HISTORY_STEP_CLASS_COUNT,
            profile_id: [0; 32],
        };
        profile.profile_id = profile.derive_id();
        profile
    }

    pub fn is_for_proof_bank(self, history_proof_bank_id: [u8; 32]) -> bool {
        self == Self::for_proof_bank(history_proof_bank_id) && self.profile_id == self.derive_id()
    }

    fn derive_id(self) -> [u8; 32] {
        poseidon2b_hash_bytes(PROFILE_ID_DOMAIN, &self.encode_parameters())
    }

    fn encode_parameters(self) -> [u8; PROFILE_BYTES] {
        let mut encoded = [0u8; PROFILE_BYTES];
        let mut cursor = 0;
        encoded[cursor..cursor + 2].copy_from_slice(&self.wire_version.to_le_bytes());
        cursor += 2;
        encoded[cursor..cursor + 32].copy_from_slice(&self.genesis_hash);
        cursor += 32;
        encoded[cursor..cursor + 32].copy_from_slice(&self.history_proof_bank_id);
        cursor += 32;
        encoded[cursor..cursor + 4].copy_from_slice(&self.max_block_bytes.to_le_bytes());
        cursor += 4;
        encoded[cursor..cursor + 4].copy_from_slice(&self.max_terminal_bytes.to_le_bytes());
        cursor += 4;
        encoded[cursor..cursor + 4].copy_from_slice(&self.max_segment_bytes.to_le_bytes());
        cursor += 4;
        encoded[cursor..cursor + 2].copy_from_slice(&self.max_header_batch.to_le_bytes());
        cursor += 2;
        encoded[cursor..cursor + 2].copy_from_slice(&self.finality_depth.to_le_bytes());
        cursor += 2;
        encoded[cursor] = self.history_terminal_version;
        cursor += 1;
        encoded[cursor] = self.history_class_count;
        encoded
    }

    fn decode_parameters(encoded: &[u8; PROFILE_BYTES], profile_id: [u8; 32]) -> Self {
        let mut cursor = 0;
        let wire_version = u16::from_le_bytes(encoded[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;
        let genesis_hash = encoded[cursor..cursor + 32].try_into().unwrap();
        cursor += 32;
        let history_proof_bank_id = encoded[cursor..cursor + 32].try_into().unwrap();
        cursor += 32;
        let max_block_bytes = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let max_terminal_bytes =
            u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let max_segment_bytes = u32::from_le_bytes(encoded[cursor..cursor + 4].try_into().unwrap());
        cursor += 4;
        let max_header_batch = u16::from_le_bytes(encoded[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;
        let finality_depth = u16::from_le_bytes(encoded[cursor..cursor + 2].try_into().unwrap());
        cursor += 2;
        let history_terminal_version = encoded[cursor];
        cursor += 1;
        let history_class_count = encoded[cursor];
        Self {
            wire_version,
            genesis_hash,
            history_proof_bank_id,
            max_block_bytes,
            max_terminal_bytes,
            max_segment_bytes,
            max_header_batch,
            finality_depth,
            history_terminal_version,
            history_class_count,
            profile_id,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkProfileRequest {
    pub expected_profile_id: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetworkProfileResponse {
    pub profile: NetworkProfile,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NetworkProfileCodec;

#[async_trait]
impl request_response::Codec for NetworkProfileCodec {
    type Protocol = StreamProtocol;
    type Request = NetworkProfileRequest;
    type Response = NetworkProfileResponse;

    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Request>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut encoded = [0u8; REQUEST_BYTES];
        io.read_exact(&mut encoded).await?;
        if encoded[..4] != REQUEST_MAGIC {
            return Err(invalid_data(
                "invalid network-profile request magic/version",
            ));
        }
        ensure_eof(io).await?;
        Ok(NetworkProfileRequest {
            expected_profile_id: encoded[4..].try_into().unwrap(),
        })
    }

    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> io::Result<Self::Response>
    where
        T: AsyncRead + Unpin + Send,
    {
        let mut encoded = [0u8; RESPONSE_BYTES];
        io.read_exact(&mut encoded).await?;
        if encoded[..4] != RESPONSE_MAGIC {
            return Err(invalid_data(
                "invalid network-profile response magic/version",
            ));
        }
        let parameters: [u8; PROFILE_BYTES] = encoded[4..4 + PROFILE_BYTES].try_into().unwrap();
        let profile_id = encoded[4 + PROFILE_BYTES..].try_into().unwrap();
        let profile = NetworkProfile::decode_parameters(&parameters, profile_id);
        if profile.derive_id() != profile.profile_id {
            return Err(invalid_data("network-profile digest mismatch"));
        }
        ensure_eof(io).await?;
        Ok(NetworkProfileResponse { profile })
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
        let mut encoded = [0u8; REQUEST_BYTES];
        encoded[..4].copy_from_slice(&REQUEST_MAGIC);
        encoded[4..].copy_from_slice(&request.expected_profile_id);
        io.write_all(&encoded).await
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
        if response.profile.derive_id() != response.profile.profile_id {
            return Err(invalid_data(
                "refusing to encode inconsistent network profile",
            ));
        }
        let mut encoded = [0u8; RESPONSE_BYTES];
        encoded[..4].copy_from_slice(&RESPONSE_MAGIC);
        encoded[4..4 + PROFILE_BYTES].copy_from_slice(&response.profile.encode_parameters());
        encoded[4 + PROFILE_BYTES..].copy_from_slice(&response.profile.profile_id);
        io.write_all(&encoded).await
    }
}

async fn ensure_eof<T: AsyncRead + Unpin>(io: &mut T) -> io::Result<()> {
    let mut trailing = [0u8; 1];
    match io.read(&mut trailing).await? {
        0 => Ok(()),
        _ => Err(invalid_data(
            "trailing bytes after fixed network-profile frame",
        )),
    }
}

fn invalid_data(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::AsyncWriteExt;
    use libp2p::request_response::Codec;

    const TEST_PROOF_BANK_ID: [u8; 32] = [0x5a; 32];

    async fn decode_response(bytes: &[u8]) -> io::Result<NetworkProfileResponse> {
        let mut codec = NetworkProfileCodec;
        let mut cursor = futures::io::Cursor::new(bytes.to_vec());
        codec
            .read_response(&StreamProtocol::new("/test"), &mut cursor)
            .await
    }

    #[tokio::test]
    async fn current_profile_round_trips_exactly() {
        let profile = NetworkProfile::for_proof_bank(TEST_PROOF_BANK_ID);
        assert!(profile.is_for_proof_bank(TEST_PROOF_BANK_ID));
        let mut codec = NetworkProfileCodec;
        let mut encoded = futures::io::Cursor::new(Vec::new());
        codec
            .write_response(
                &StreamProtocol::new("/test"),
                &mut encoded,
                NetworkProfileResponse { profile },
            )
            .await
            .unwrap();
        encoded.flush().await.unwrap();
        let decoded = decode_response(encoded.get_ref()).await.unwrap();
        assert_eq!(decoded.profile, profile);
    }

    #[tokio::test]
    async fn altered_profile_and_trailing_bytes_fail_closed() {
        let profile = NetworkProfile::for_proof_bank(TEST_PROOF_BANK_ID);
        let mut codec = NetworkProfileCodec;
        let mut encoded = futures::io::Cursor::new(Vec::new());
        codec
            .write_response(
                &StreamProtocol::new("/test"),
                &mut encoded,
                NetworkProfileResponse { profile },
            )
            .await
            .unwrap();
        let mut wrong = encoded.into_inner();
        wrong[10] ^= 1;
        assert!(decode_response(&wrong).await.is_err());

        let profile = NetworkProfile::for_proof_bank(TEST_PROOF_BANK_ID);
        let mut canonical = futures::io::Cursor::new(Vec::new());
        codec
            .write_response(
                &StreamProtocol::new("/test"),
                &mut canonical,
                NetworkProfileResponse { profile },
            )
            .await
            .unwrap();
        let mut trailing = canonical.into_inner();
        trailing.push(0);
        assert!(decode_response(&trailing).await.is_err());
    }

    /// The v1.2 fork must partition the network **at its activation height**
    /// and nowhere else. This profile is exchanged before a connection becomes
    /// usable, so raising either of these two fields here would partition at
    /// *installation* instead: an upgraded node and a legacy node would refuse
    /// each other weeks before the fork, and the upgrade could never spread.
    ///
    /// Both therefore stay on the pre-activation baseline. Height-selected
    /// consensus admits the wider terminal and the shared-path encoding on its
    /// own, at the height where every node switches together.
    #[test]
    fn the_advertised_profile_stays_on_the_pre_activation_baseline() {
        use jetsam_chain::consensus::wire_limits::{
            MAX_HISTORY_STEP_TERMINAL_TRANSPORT_BYTES, V1_MAX_HISTORY_STEP_TERMINAL_BYTES,
        };
        use jetsam_chain::history_step::HISTORY_STEP_TERMINAL_SHARED_PATH_VERSION;

        let profile = NetworkProfile::for_proof_bank(TEST_PROOF_BANK_ID);
        assert_eq!(profile.wire_version, 7);
        assert_eq!(
            profile.max_terminal_bytes,
            V1_MAX_HISTORY_STEP_TERMINAL_BYTES as u32
        );
        assert_eq!(profile.max_terminal_bytes, 1_048_576);
        assert_eq!(profile.history_terminal_version, 4);
        assert_ne!(
            profile.history_terminal_version,
            HISTORY_STEP_TERMINAL_SHARED_PATH_VERSION
        );
        assert!(
            MAX_HISTORY_STEP_TERMINAL_TRANSPORT_BYTES > profile.max_terminal_bytes as usize,
            "this binary allocates for the raised cap while advertising the old \
             one: that headroom is exactly what lets it stay connected to \
             legacy peers until the activation height"
        );
        // Anchored digest of every advertised field for a fixed bank id. Any
        // change to a profile field — not just the two above — moves this and
        // partitions the network at installation.
        assert_eq!(
            profile.profile_id,
            pre_activation_profile_id(),
            "the advertised network profile is no longer the one live peers speak"
        );
    }

    /// The anchored profile id of the network **this build is linked against**.
    ///
    /// # Why it is chosen by the genesis and not by a `cfg`
    ///
    /// Every field the digest covers is shared by the two chains except
    /// `genesis_hash`, and that one must differ: it is what makes a node of one
    /// chain refuse a node of the other at the profile exchange, before a block
    /// is ever offered. So there are two anchors, and one of them has to be
    /// picked.
    ///
    /// `jetsam_p2p` declares no `testnet` feature — the profile is a property of
    /// the `jetsam_chain` it links, not of this crate — so
    /// `cfg(feature = "testnet")` here is always false and would silently pick
    /// the public network's anchor inside a test-chain build. Reading the
    /// genesis cannot drift that way: it is the very input that makes the two
    /// digests differ, and an unknown one stops the test instead of passing it.
    fn pre_activation_profile_id() -> [u8; 32] {
        /// Profile id for `TEST_PROOF_BANK_ID` under the pre-activation
        /// baseline, i.e. what v1.1.2 nodes on the live public chain advertise:
        /// `e05175deb0ddd4592d7cca7f708f9ff63b97852c5750c2a9c4ab80a4224ddf3d`.
        const MAINNET: [u8; 32] = [
            0xe0, 0x51, 0x75, 0xde, 0xb0, 0xdd, 0xd4, 0x59,
            0x2d, 0x7c, 0xca, 0x7f, 0x70, 0x8f, 0x9f, 0xf6,
            0x3b, 0x97, 0x85, 0x2c, 0x57, 0x50, 0xc2, 0xa9,
            0xc4, 0xab, 0x80, 0xa4, 0x22, 0x4d, 0xdf, 0x3d,
        ];
        /// The same digest over the test chain's genesis:
        /// `8a84d9a6f9440c1c74c432d0c5dd937754b554978d2bda7727f22a2d998c03f3`.
        ///
        /// Captured from this source tree, which is the source the nodes of that
        /// chain run — so it is the profile those peers speak. It has never been
        /// read off the wire: the exchange happens inside the Noise session and
        /// the node prints the id at `debug` only. What it guarantees is what
        /// the public anchor guarantees: no later edit to an advertised field
        /// moves the id without a test saying so.
        const TESTNET: [u8; 32] = [
            0x8a, 0x84, 0xd9, 0xa6, 0xf9, 0x44, 0x0c, 0x1c,
            0x74, 0xc4, 0x32, 0xd0, 0xc5, 0xdd, 0x93, 0x77,
            0x54, 0xb5, 0x54, 0x97, 0x8d, 0x2b, 0xda, 0x77,
            0x27, 0xf2, 0x2a, 0x2d, 0x99, 0x8c, 0x03, 0xf3,
        ];
        /// `jetsam_chain::consensus::genesis`, anchored there by
        /// `genesis_block_id_is_canonical`.
        const MAINNET_GENESIS_ID: [u8; 32] = [
            0x6e, 0x59, 0x2c, 0x07, 0xbe, 0x6f, 0xd1, 0xb4,
            0x25, 0x9e, 0xea, 0xcb, 0xf4, 0xeb, 0x7e, 0xb2,
            0x94, 0x8a, 0x77, 0xf1, 0xd0, 0x26, 0x26, 0xa1,
            0x2f, 0xda, 0xb4, 0x2c, 0x44, 0x8c, 0x5f, 0x44,
        ];
        /// Anchored there by
        /// `testnet_genesis_block_id_is_canonical_and_differs_from_mainnet`.
        const TESTNET_GENESIS_ID: [u8; 32] = [
            0xb3, 0xef, 0xb3, 0xc1, 0xd3, 0x1f, 0xee, 0x8b,
            0x9a, 0xee, 0x7b, 0x06, 0xcb, 0x11, 0x2f, 0xae,
            0xa5, 0xab, 0xc1, 0xce, 0xb7, 0x35, 0xd2, 0x1c,
            0xa5, 0xa3, 0x90, 0x1b, 0x11, 0x0f, 0x99, 0x6d,
        ];

        let genesis = block_id(&genesis_header());
        match genesis {
            MAINNET_GENESIS_ID => MAINNET,
            TESTNET_GENESIS_ID => TESTNET,
            other => panic!(
                "this build speaks a network with genesis {}, which has no anchored \
                 network profile: add one rather than leaving the exchange untested",
                hex_lower(&other)
            ),
        }
    }

    fn hex_lower(bytes: &[u8; 32]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    #[test]
    fn proof_bank_identity_is_part_of_the_network_profile() {
        let first = NetworkProfile::for_proof_bank([1; 32]);
        let second = NetworkProfile::for_proof_bank([2; 32]);
        assert_ne!(first.profile_id, second.profile_id);
        assert!(!first.is_for_proof_bank([2; 32]));
    }
}
