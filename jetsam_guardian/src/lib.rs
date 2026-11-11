// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The Jetsam v1.7 guardian software (plan T5.1-T5.6, design
//! `jetsam-v17-gardien-logiciel_DESIGN.md`).
//!
//! Four programs run on a guardian machine, each with its own trust boundary:
//!
//! * `jg-observe` reads **its** node only (loopback RPC, an allow-list of
//!   methods) and applies the signing policy SG-1..SG-9 and SG-17
//!   ([`observer`]);
//! * `jg-net` is the only one that talks to the network: the other sources
//!   (SG-10, [`witness`]), the proposals, the publication of the signed feed
//!   ([`feed`]); it holds no signing secret;
//! * `jg-sign` holds the share and the journal, in an EMPTY network
//!   namespace; it re-does every safety computation itself (SG-11..SG-16,
//!   [`signer`], [`journal`]) and has a single signing path;
//! * `jg-aggregate` combines any 5 valid partials into the group signature
//!   ([`aggregator`]); anyone may run it.
//!
//! Keys: [`ceremony`] (offline dealing, Shamir 5 of 7, Feldman commitments,
//! re-sharing at a constant group key, revocation certificate) and [`seal`]
//! (a share at rest is sealed for its machine key, itself sealed by the
//! operator's passphrase).
//!
//! The statement (416 bytes), its message, the DST, the equivocation body and
//! the anchor leaf are the node's own code (`jetsam_chain::consensus::guardian`),
//! linked from the same commit: nothing of the format is redefined here.

pub mod activations;
pub mod aggregator;
pub mod base_capture;
pub mod ceremony;
pub mod chain;
pub mod checkpoint;
pub mod config;
pub mod fast;
pub mod fast_release;
pub mod fast_run;
pub mod feed;
pub mod fill;
pub mod harden;
pub mod ipc;
pub mod journal;
pub mod l1;
pub mod l1tx;
pub mod light_feed;
pub mod observer;
pub mod onion_feed;
pub mod relay;
pub mod relayer;
pub mod rpc;
pub mod seal;
pub mod signer;
#[doc(hidden)]
pub mod testkit;
pub mod threshold;
pub mod u256;
pub mod watch;
pub mod witness;

pub use jetsam_chain::consensus::guardian::{
    GuardianStatement, GUARDIAN_DST, GUARDIAN_STATEMENT_DOMAIN, GUARDIAN_STATEMENT_VERSION,
    SIGNATURE_BYTES, STATEMENT_BYTES,
};

/// The block id of the mainnet genesis: the `network` of a mainnet guardian
/// (anchored in `jetsam_chain`, `genesis_block_id_is_canonical`; it never moves).
pub const MAINNET_NETWORK: [u8; 32] = [
    0x6e, 0x59, 0x2c, 0x07, 0xbe, 0x6f, 0xd1, 0xb4, 0x25, 0x9e, 0xea, 0xcb, 0xf4, 0xeb, 0x7e, 0xb2, 0x94, 0x8a,
    0x77, 0xf1, 0xd0, 0x26, 0x26, 0xa1, 0x2f, 0xda, 0xb4, 0x2c, 0x44, 0x8c, 0x5f, 0x44,
];

/// The block id of the test chain's genesis (anchored in `jetsam_chain`,
/// `testnet_genesis_block_id_is_canonical_and_differs_from_mainnet`): the
/// `network` of a public test network guardian, and also of the private
/// rehearsal's, whose builds (`rehearsal-v1-7`) are made from the test chain
/// profile and share its genesis.
pub const TESTNET_NETWORK: [u8; 32] = [
    0xd9, 0xd1, 0x56, 0xee, 0x35, 0xe7, 0x25, 0xcf, 0x6b, 0x13, 0x32, 0x16, 0x80, 0x9c, 0x1b, 0x0f, 0x16, 0x5d,
    0x11, 0x20, 0xc9, 0xcf, 0x42, 0xcc, 0x2b, 0xfe, 0xb0, 0xef, 0x90, 0x31, 0xee, 0x85,
];

/// The private rehearsal's L1, a local devnet: `GUARDIAN_L1_CHAIN_ID` of the
/// `rehearsal-v1-7` builds (`jetsam_chain::consensus::guardian`). A public
/// network's guardians sign for another L1 (1 on the mainnet, 11 155 111 on the
/// test network), and its nodes refuse a statement naming this one.
pub const REHEARSAL_L1_CHAIN_ID: u64 = 31_337;

/// The rules a guardian program starts under (D66), from the domain it signs
/// for. The mainnet and the public test network have the same strict rules:
/// SG-4 floor, every source through Tor or on the loopback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profile {
    /// The mainnet network (its genesis id).
    Mainnet,
    /// The public test network: the test chain's genesis, with any L1 domain
    /// but the rehearsal's.
    PublicTestnet,
    /// Any other network whose guardians sign for an L1 other than the
    /// rehearsal's: strict as well (a public test chain relaunched with a new
    /// genesis is never taken for a rehearsal).
    OtherNetwork,
    /// The private rehearsal: the rehearsal's L1 (chain 31 337) on a network
    /// that is not the mainnet; on the test chain's genesis, which it shares
    /// with the public test network, its synthetic bridge as well.
    PrivateRehearsal,
}

impl Profile {
    /// The profile of a guardian signing for `domain`.
    pub fn of(domain: &Domain) -> Self {
        let rehearsal_l1 = domain.l1_chain_id == REHEARSAL_L1_CHAIN_ID;
        if domain.network == MAINNET_NETWORK {
            Profile::Mainnet
        } else if domain.network == TESTNET_NETWORK {
            let bridge = jetsam_chain::consensus::destination::state::REHEARSAL_L1_BRIDGE.address;
            if rehearsal_l1 && domain.l1_bridge == bridge {
                Profile::PrivateRehearsal
            } else {
                Profile::PublicTestnet
            }
        } else if rehearsal_l1 {
            Profile::PrivateRehearsal
        } else {
            Profile::OtherNetwork
        }
    }

    /// The public networks' rules apply (everything but the private rehearsal).
    pub fn strict(self) -> bool {
        self != Profile::PrivateRehearsal
    }
}

impl std::fmt::Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Profile::Mainnet => "the mainnet network",
            Profile::PublicTestnet => "the public test network",
            Profile::OtherNetwork => "a network that is not the private rehearsal's",
            Profile::PrivateRehearsal => "the private rehearsal",
        })
    }
}

/// The statements `jg-net` evaluates per 15 s round, the highest first:
/// the proposals of the publication points and the attestations due.
pub const NET_ROUND_CANDIDATES: usize = 32;

/// keccak-256, the node's.
pub fn keccak256(data: &[u8]) -> [u8; 32] {
    jetsam_chain::consensus::destination::mpt::keccak256(data)
}

/// Lower-case hex without prefix.
pub fn hex32(bytes: &[u8]) -> String {
    hex::encode(bytes)
}

/// Parse 32 bytes of hex (with or without `0x`).
pub fn parse32(text: &str) -> anyhow::Result<[u8; 32]> {
    let bytes = hex::decode(text.trim().trim_start_matches("0x"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected 32 bytes of hex"))
}

/// The domain words a guardian signs under (words 0-4 of a statement):
/// fixed by its configuration, compared by the signer to every statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Domain {
    /// Word 1: the Jetsam network (block id of the profile's genesis).
    pub network: [u8; 32],
    /// Word 2: the L1 chain id.
    pub l1_chain_id: u64,
    /// Word 3: the L1 bridge contract.
    pub l1_bridge: [u8; 20],
}

impl Domain {
    /// Words 0-4 match this domain and the schema.
    pub fn matches(&self, s: &GuardianStatement) -> bool {
        s.domain == GUARDIAN_STATEMENT_DOMAIN
            && s.network == self.network
            && s.l1_chain_id == jetsam_chain::consensus::guardian::u256_word(self.l1_chain_id)
            && s.l1_bridge == self.l1_bridge
            && s.schema_version == GUARDIAN_STATEMENT_VERSION
    }
}

/// Seconds since the Unix epoch, on this machine's clock.
pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A5-14: write `bytes` to `path` so that a stop or a power cut leaves either the previous file or the new one,
/// whole: a temporary beside it under its own name (`<file name>.tmp`, replaced if a crash left one), synced
/// (`sync_all`), renamed over `path`, then its directory synced (the rename itself is on disk when it returns).
pub fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;
    let name = path.file_name().ok_or_else(|| anyhow::anyhow!("{}: not a file path", path.display()))?;
    let mut tmp_name = name.to_os_string();
    tmp_name.push(".tmp");
    let tmp = path.with_file_name(tmp_name);
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d,
        _ => std::path::Path::new("."),
    };
    let mut f = std::fs::File::create(&tmp).map_err(|e| anyhow::anyhow!("{}: {e}", tmp.display()))?;
    f.write_all(bytes).map_err(|e| anyhow::anyhow!("{}: {e}", tmp.display()))?;
    f.sync_all().map_err(|e| anyhow::anyhow!("{}: sync: {e}", tmp.display()))?;
    drop(f);
    std::fs::rename(&tmp, path).map_err(|e| anyhow::anyhow!("{} -> {}: {e}", tmp.display(), path.display()))?;
    std::fs::File::open(dir).and_then(|d| d.sync_all()).map_err(|e| anyhow::anyhow!("{}: sync: {e}", dir.display()))?;
    Ok(())
}
