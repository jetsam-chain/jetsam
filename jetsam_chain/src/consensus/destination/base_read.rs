// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! v1.7, the return path over Base (D94, D101): from a finalized L1 execution
//! state root to a Base state root, then to the bridge's `withdrawals[n]`.
//!
//! The rule follows Base's own finality, as Base defines it. A dispute game of
//! the pinned type (621, `AggregateVerifier` 0.3.0, base/contracts v8.4.0) is
//! read only when, at the finalized L1 block, the equivalent of
//! `AnchorStateRegistry.isGameClaimValid(game)` (registry 3.7.0) holds:
//!
//! 1. *proper*: registered by the factory (`games(type, rootClaim, extraData)`
//!    is the game, whose registry is this one), not blacklisted, not retired
//!    (`createdAt > retirementTimestamp`), the system not paused
//!    (`SuperchainConfig.paused(0) || paused(portal)`, each a pause younger
//!    than its expiry);
//! 2. *respected*: the game type was the respected one when it was created;
//! 3. *finalized*: resolved (`resolvedAt ≠ 0`, a final status) and
//!    `block.timestamp − resolvedAt > disputeGameFinalityDelaySeconds`;
//! 4. resolved `DEFENDER_WINS`.
//!
//! Its root claim, an output root of version 0
//! (`keccak(0 ‖ stateRoot ‖ messagePasserStorageRoot ‖ blockHash)`), then
//! opens to the Base state root, under which the bridge is read exactly as on
//! the L1 ([`super::l1_read`]): its code hash, its `genesisId`, then
//! `withdrawals[n] = keccak(owner ‖ amount)`.
//!
//! Every value is a storage word proven under the L1 root; the game's root
//! claim and L2 block are its clone's immutable arguments, proven by its code
//! hash (the code is rebuilt from a pinned implementation and the arguments).
//! What gives the slots their meaning is pinned: the implementation of the
//! registry, of the factory, of `SystemConfig` and of `SuperchainConfig`
//! (their EIP-1967 slot), and the game implementations. Any upgrade of one of
//! them by Base is refused until a Jetsam release reads it: the return waits,
//! it never trusts a layout it has not read.
//!
//! Pure functions, not yet wired into the consensus. The L1 time `l1_timestamp`
//! is the timestamp of the execution block whose state root is read (its
//! header's, [`super::mpt::execution_header_timestamp`]): under Gloas it is not
//! the finalized beacon slot's.

use super::l1_read::{
    read_account, read_storage_word, verify_bridge_account, verify_withdrawal, L1Account,
    L1BridgeParams, L1ReadError,
};
use super::mpt::keccak256;
use super::ssz::Root;

/// An L1 account address.
pub type Address = [u8; 20];
/// A Merkle-Patricia proof: its nodes, root first.
pub type Nodes = Vec<Vec<u8>>;

/// `GameStatus.IN_PROGRESS`.
pub const GAME_STATUS_IN_PROGRESS: u8 = 0;
/// `GameStatus.CHALLENGER_WINS`.
pub const GAME_STATUS_CHALLENGER_WINS: u8 = 1;
/// `GameStatus.DEFENDER_WINS`.
pub const GAME_STATUS_DEFENDER_WINS: u8 = 2;

/// EIP-1967 implementation slot of a proxy:
/// `keccak("eip1967.proxy.implementation") − 1`.
pub const IMPLEMENTATION_SLOT: Root =
    h32("360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc");

/// `SystemConfig.OPTIMISM_PORTAL_SLOT`: `keccak("systemconfig.optimismportal") − 1`.
pub const OPTIMISM_PORTAL_SLOT: Root =
    h32("4b6c74f9e688cb39801f2112c14a8c57232a3fc5202e1444126d4bce86eb19ac");

// Storage layouts (`snapshots/storageLayout/*.json` of base/contracts v8.4.0,
// checked against the deployed values): the registry holds `systemConfig` at
// slot 0 offset 2, `disputeGameFactory` at 1, the `disputeGameBlacklist`
// mapping at 5, `respectedGameType` (offset 0) and `retirementTimestamp`
// (offset 4) at 6; the factory its `_disputeGames` mapping (`Hash => GameId`)
// at 103; `SystemConfig` its `superchainConfig` at 108 (offset 0);
// `SuperchainConfig` its `pauseTimestamps` mapping at 0; a game, at slot 0,
// `createdAt` (offset 0, 8 B), `resolvedAt` (8, 8 B), `status` (16),
// `initialized` (17), `wasRespectedGameTypeWhenCreated` (18).
const ASR_SYSTEM_CONFIG_SLOT: u64 = 0;
const ASR_FACTORY_SLOT: u64 = 1;
const ASR_BLACKLIST_MAPPING: u64 = 5;
const ASR_RETIREMENT_SLOT: u64 = 6;
const FACTORY_GAMES_MAPPING: u64 = 103;
const SYSTEM_CONFIG_SUPERCHAIN_SLOT: u64 = 108;
const SUPERCHAIN_PAUSE_MAPPING: u64 = 0;
const GAME_STATE_SLOT: u64 = 0;

/// Order of the registry's slot proofs in [`GameProof::anchor_state_registry`].
pub const ASR_SLOT_SYSTEM_CONFIG: usize = 0;
pub const ASR_SLOT_FACTORY: usize = 1;
pub const ASR_SLOT_BLACKLIST: usize = 2;
pub const ASR_SLOT_RETIREMENT: usize = 3;
pub const ASR_SLOT_IMPLEMENTATION: usize = 4;

/// Offset of the arguments in a clone's code, and bytes of arguments before
/// the intermediate roots: creator (20), root claim (32), L1 head (32), L2
/// block number (32), parent (20).
const CLONE_ARGS_OFFSET: usize = 0x62;
const ARGS_FIXED_BYTES: usize = 0x88;

/// The L1 contracts of one Base network and what gives their storage its
/// meaning. Consensus constants on a real network; injected so that the rule
/// can be exercised on vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseParams {
    pub anchor_state_registry: Address,
    pub anchor_state_registry_impl: Address,
    pub dispute_game_factory: Address,
    pub dispute_game_factory_impl: Address,
    pub system_config: Address,
    pub system_config_impl: Address,
    pub superchain_config: Address,
    pub superchain_config_impl: Address,
    /// The game type (621); every implementation below is of this type.
    pub game_type: u32,
    /// The game implementations read (their storage layout, their registry
    /// and factory, their delays are this module's).
    pub game_impls: &'static [Address],
    /// `SLOW_BLOCK_INTERVAL / SLOW_INTERMEDIATE_BLOCK_INTERVAL` of those
    /// implementations: the intermediate roots in a game's arguments.
    pub intermediate_roots: usize,
    /// `DISPUTE_GAME_FINALITY_DELAY_SECONDS` of the registry implementation.
    pub finality_delay_seconds: u64,
    /// `PAUSE_EXPIRY` of the `SuperchainConfig` implementation.
    pub pause_expiry_seconds: u64,
    /// FIXRC RC-3 (AUDIT-FINAL2): later pins, each in force from its height
    /// on (an update of Base's contracts re-pinned by a release of the nodes
    /// at a height): the blocks below keep the pins of their time, so the
    /// history still replays. Empty: these pins for ever. An entry's own
    /// table is not read (the pins in force keep this one: [`Self::at`] of the
    /// same height gives them again).
    pub rotations: &'static [(u64, BaseParams)],
}

impl BaseParams {
    /// The pins in force at `height`: the last rotation at or below it, else
    /// these. Only rotations above the return-over-Base height are meant;
    /// the first pins must be the base ones.
    pub const fn at(&self, height: u64) -> Self {
        let mut found = *self;
        let mut i = 0;
        while i < self.rotations.len() {
            if self.rotations[i].0 <= height {
                found = BaseParams {
                    rotations: self.rotations,
                    ..self.rotations[i].1
                };
            }
            i += 1;
        }
        found
    }
}

/// Base's main network, read on the L1 on 2026-11-09 (block 26 379 491):
/// registry 3.7.0, factory 1.5.0, `SystemConfig` 3.14.0, `SuperchainConfig`
/// 2.5.0, `AggregateVerifier` 0.3.0 (source identical to v8.4.0).
pub const BASE_MAINNET: BaseParams = BaseParams {
    anchor_state_registry: h20("909f6cf47ed12f010a796527f562bfc26c7f4e72"),
    anchor_state_registry_impl: h20("4483f964f6711cb55f633820ed174e780369b99d"),
    dispute_game_factory: h20("43edb88c4b80fdd2adff2412a7bebf9df42cb40e"),
    dispute_game_factory_impl: h20("94eab8f73fd83407d26c006d159923760d483029"),
    system_config: h20("73a79fab69143498ed3712e519a88a918e1f4072"),
    system_config_impl: h20("56ca4f4f5ba9b42d360b48429e8c4a60c5a0795d"),
    superchain_config: h20("b535ff7f118260a952ce65e7ff41b1743de8ee6c"),
    superchain_config_impl: h20("c4ceeb264fa35a022e470acc3c6a522308f3ec02"),
    game_type: 621,
    game_impls: &[h20("c08b8ffaf04abb1d7013b74a0e57eccbd7fa5a70")],
    intermediate_roots: 20,
    finality_delay_seconds: 0,
    pause_expiry_seconds: 7_884_000,
    rotations: &[],
};

/// Base Sepolia, read on Sepolia on 2026-11-09 (block 12 099 618): the same
/// versions (`SystemConfig` 3.14.0+max-gas-limit-2000M).
pub const BASE_SEPOLIA: BaseParams = BaseParams {
    anchor_state_registry: h20("2ff5cc82dbf333ea30d8ee462178ab1707315355"),
    anchor_state_registry_impl: h20("b1cc9f8422042eda9eb36a408002517d7c772ac7"),
    dispute_game_factory: h20("d6e6dbf4f7ea0ac412fd8b65ed297e64bb7a06e1"),
    dispute_game_factory_impl: h20("9cacf701ef24b8339cd1fdecadde7c080373fb29"),
    system_config: h20("f272670eb55e895584501d564afeb048bed26194"),
    system_config_impl: h20("aeb75e200b9bcf9343ff1a2d2b1b3976e7339ca4"),
    superchain_config: h20("e4401eb53ae90a5335a51fe1828d7becf7a63508"),
    superchain_config_impl: h20("99fcf11772af0a9cc411af3cb4311a387dd55b15"),
    game_type: 621,
    game_impls: &[h20("4234216f1648a8119c6e41e6c0c73956ecc850af")],
    intermediate_roots: 20,
    finality_delay_seconds: 0,
    pause_expiry_seconds: 7_884_000,
    rotations: &[],
};

/// One contract read under a state root: its account proof, then the proofs
/// of its slots in the order the rule reads them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractProof {
    pub account: Nodes,
    pub slots: Vec<Nodes>,
}

/// Everything that proves one game's validity under an L1 state root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameProof {
    /// The game (a clone of a pinned implementation).
    pub game: Address,
    /// Its immutable arguments: creator, root claim, L1 head, L2 block
    /// number, parent, then the intermediate roots.
    pub args: Vec<u8>,
    /// The registry: `systemConfig`, `disputeGameFactory`, the blacklist
    /// entry of the game, the retirement word, the implementation.
    pub anchor_state_registry: ContractProof,
    /// The factory: the game's `GameId`, the implementation.
    pub dispute_game_factory: ContractProof,
    /// The game: its slot 0.
    pub game_account: ContractProof,
    /// `SystemConfig`: `superchainConfig`, the portal, the implementation.
    pub system_config: ContractProof,
    /// `SuperchainConfig`: `pauseTimestamps[0]`, `pauseTimestamps[portal]`,
    /// the implementation.
    pub superchain_config: ContractProof,
}

/// What the proofs show of a game and of the contracts that judge it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameFacts {
    pub game: Address,
    /// The implementation its code clones.
    pub game_impl: Address,
    pub root_claim: Root,
    pub l2_block_number: u64,
    pub created_at: u64,
    pub resolved_at: u64,
    pub status: u8,
    /// `initialized` (offset 17 of slot 0).
    pub initialized: bool,
    pub was_respected: bool,
    /// The two words the fast lane reads (slots 5 and 7), when the game's
    /// proof carries them ([`FAST_GAME_SLOTS`]); `None` otherwise.
    pub fast: Option<FastGameFacts>,
    /// The proxy the factory registered for the game's UUID (zero if none).
    pub registered_proxy: Address,
    pub blacklisted: bool,
    pub retirement_timestamp: u64,
    pub asr_system_config: Address,
    pub asr_dispute_game_factory: Address,
    pub superchain_config: Address,
    /// The pause identifier `SystemConfig.optimismPortal()`.
    pub optimism_portal: Address,
    /// `pauseTimestamps[0]` and `pauseTimestamps[portal]` (uint256).
    pub pause_global: Root,
    pub pause_portal: Root,
    /// The four proxies' EIP-1967 implementations: registry, factory,
    /// `SystemConfig`, `SuperchainConfig`.
    pub implementations: [Address; 4],
}

/// What the fast lane reads of a game beside its slot 0 (D127, RETOUR-RAPIDE
/// SPEC §3): `counteredByIntermediateRootIndexPlusOne` (slot 5, uint256, 0 =
/// not countered) and `proofCount` (slot 7, offset 8, one byte; offset 0 holds
/// `expectedResolution`). Layout of `AggregateVerifier` 0.3.0 (Clone and
/// Solady's ReentrancyGuard hold no sequential slot): 0 the clocks and flags,
/// 1-2 `startingOutputRoot`, 3 `bondRecipient` and its flags, 4 `bondAmount`,
/// 5 `counteredBy…`, 6 `proofTypeToProver`, 7 `expectedResolution` and
/// `proofCount`, 8 `scheduleId`; each word checked against its getter on two
/// real games of Base Sepolia, 2026-11-10.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastGameFacts {
    pub proof_count: u8,
    pub countered_by: Root,
}

/// The slots of a game the fast lane reads, in the order of its proof: the
/// state word, `counteredByIntermediateRootIndexPlusOne`, the word holding
/// `proofCount`. The slow return reads slot 0 alone.
pub const FAST_GAME_SLOTS: [u64; 3] = [GAME_STATE_SLOT, GAME_COUNTERED_SLOT, GAME_PROOF_COUNT_SLOT];
const GAME_COUNTERED_SLOT: u64 = 5;
const GAME_PROOF_COUNT_SLOT: u64 = 7;

/// A game Base holds valid at the L1 block read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValidGame {
    pub game: Address,
    pub l2_block_number: u64,
    pub root_claim: Root,
}

/// A Base state root that Base itself holds final at the L1 block read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseState {
    pub game: Address,
    pub l2_block_number: u64,
    pub state_root: Root,
}

/// The preimage of an output root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutputRootPreimage {
    pub version: Root,
    pub state_root: Root,
    pub message_passer_storage_root: Root,
    pub block_hash: Root,
}

/// One release read under a Base state root (rule V7 there).
#[derive(Clone, Copy, Debug)]
pub struct BridgeRead<'a> {
    pub bridge: &'a L1BridgeParams,
    pub account_proof: &'a [Vec<u8>],
    pub genesis_proof: &'a [Vec<u8>],
    pub n: u64,
    pub owner: &'a [u8; 32],
    pub amount: u64,
    pub storage_proof: &'a [Vec<u8>],
}

/// A contract the rule pins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseContract {
    AnchorStateRegistry,
    DisputeGameFactory,
    SystemConfig,
    SuperchainConfig,
}

/// Why a game is not read as valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseGameError {
    /// A proof does not open under its root, or an account is absent.
    Read(L1ReadError),
    /// A contract proof without the slots the rule reads, or arguments of
    /// another length than the pinned implementations'.
    ProofShape,
    /// The game's code is no clone of a pinned implementation over these
    /// arguments (another implementation, or arguments not the game's).
    UnknownGameCode,
    /// The L2 block number does not fit 64 bits.
    L2BlockOverflow,
    /// A proxy's implementation is not the pinned one: Base upgraded it.
    UnknownImplementation(BaseContract),
    /// The registry's `systemConfig` or `disputeGameFactory`, or the
    /// `SystemConfig`'s `superchainConfig`, is not the pinned contract.
    Wiring(BaseContract),
    NotRegistered,
    Blacklisted,
    Retired,
    Paused,
    NotRespected,
    NotResolved,
    NotFinalized,
    NotDefenderWins,
    /// The fast lane: the game is resolved (no longer `IN_PROGRESS`).
    NotInProgress,
    /// The fast lane: the game is not initialized.
    NotInitialized,
    /// The fast lane: the proof does not carry slots 5 and 7.
    FastFactsMissing,
    /// The fast lane: no proof was verified for the game (`proofCount == 0`).
    NoProof,
    /// The fast lane: the game was countered (a challenge proved one of its
    /// intermediate roots wrong).
    Countered,
}

impl From<L1ReadError> for BaseGameError {
    fn from(error: L1ReadError) -> Self {
        Self::Read(error)
    }
}

/// Why a Base state root or a release is not proven.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseReturnError {
    Game(BaseGameError),
    /// The output root's version is not 0.
    OutputVersion,
    /// The preimage does not hash to the game's root claim.
    OutputRootMismatch,
    /// The bridge read under the Base state root (V7).
    Bridge(L1ReadError),
}

impl From<BaseGameError> for BaseReturnError {
    fn from(error: BaseGameError) -> Self {
        Self::Game(error)
    }
}

const fn nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        _ => panic!("lowercase hex"),
    }
}

const fn h20(s: &str) -> Address {
    let b = s.as_bytes();
    assert!(b.len() == 40);
    let mut out = [0u8; 20];
    let mut i = 0;
    while i < 20 {
        out[i] = (nibble(b[2 * i]) << 4) | nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

const fn h32(s: &str) -> Root {
    let b = s.as_bytes();
    assert!(b.len() == 64);
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = (nibble(b[2 * i]) << 4) | nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

fn u256(x: u64) -> Root {
    let mut out = [0u8; 32];
    out[24..].copy_from_slice(&x.to_be_bytes());
    out
}

fn padded(address: &Address) -> Root {
    let mut out = [0u8; 32];
    out[12..].copy_from_slice(address);
    out
}

/// The slot of `mapping[key]` for a mapping at `slot`: `keccak(key ‖ u256(slot))`.
fn mapping_slot(key: &Root, slot: u64) -> Root {
    let mut preimage = [0u8; 64];
    preimage[..32].copy_from_slice(key);
    preimage[32..].copy_from_slice(&u256(slot));
    keccak256(&preimage)
}

/// The `len` bytes at byte `offset` of a storage word (offset 0 = the
/// lowest-order byte, as Solidity packs).
fn field(word: &Root, offset: usize, len: usize) -> &[u8] {
    &word[32 - offset - len..32 - offset]
}

fn field_u64(word: &Root, offset: usize, len: usize) -> u64 {
    field(word, offset, len)
        .iter()
        .fold(0u64, |acc, b| (acc << 8) | u64::from(*b))
}

fn field_address(word: &Root, offset: usize) -> Address {
    field(word, offset, 20).try_into().expect("20 bytes")
}

/// A clone with immutable arguments (Solady `LibClone`, the factory's
/// `cloneDeterministic(impl, args, uuid)`): 98 bytes of code delegating to
/// `implementation`, the arguments, then their length + 2 (`uint16`).
pub fn clone_code(implementation: &Address, args: &[u8]) -> Vec<u8> {
    const HEAD: [u8; 54] = {
        let a = h32("36602c57343d527f9e4ac34f21c619cefc926c8bd93b54bf5a39c7ab2127a895");
        let b = h32("af1cc0691d7e3dff593da1005b363d3d373d3d3d3d6100000000000000000000");
        let mut out = [0u8; 54];
        let mut i = 0;
        while i < 54 {
            out[i] = if i < 32 { a[i] } else { b[i - 32] };
            i += 1;
        }
        out
    };
    const MID: [u8; 9] = [0x80, 0x60, 0x62, 0x36, 0x39, 0x36, 0x01, 0x3d, 0x73];
    const TAIL: [u8; 13] = [
        0x5a, 0xf4, 0x3d, 0x3d, 0x93, 0x80, 0x3e, 0x60, 0x60, 0x57, 0xfd, 0x5b, 0xf3,
    ];
    let extra = u16::try_from(args.len() + 2)
        .expect("arguments fit a clone")
        .to_be_bytes();
    let mut code = Vec::with_capacity(CLONE_ARGS_OFFSET + args.len() + 2);
    code.extend_from_slice(&HEAD);
    code.extend_from_slice(&extra);
    code.extend_from_slice(&MID);
    code.extend_from_slice(implementation);
    code.extend_from_slice(&TAIL);
    debug_assert_eq!(code.len(), CLONE_ARGS_OFFSET);
    code.extend_from_slice(args);
    code.extend_from_slice(&extra);
    code
}

/// The factory's game UUID: `keccak(abi.encode(gameType, rootClaim, extraData))`.
pub fn game_uuid(game_type: u32, root_claim: &Root, extra_data: &[u8]) -> Root {
    let padding = (32 - extra_data.len() % 32) % 32;
    let mut enc = Vec::with_capacity(128 + extra_data.len() + padding);
    enc.extend_from_slice(&u256(u64::from(game_type)));
    enc.extend_from_slice(root_claim);
    enc.extend_from_slice(&u256(0x60));
    enc.extend_from_slice(&u256(extra_data.len() as u64));
    enc.extend_from_slice(extra_data);
    enc.resize(enc.len() + padding, 0);
    keccak256(&enc)
}

/// The account at `address` and its slot words, in order.
fn read_contract(
    state_root: Root,
    address: &Address,
    proof: &ContractProof,
    slots: &[Root],
) -> Result<(L1Account, Vec<Root>), BaseGameError> {
    if proof.slots.len() != slots.len() {
        return Err(BaseGameError::ProofShape);
    }
    let account = read_account(state_root, address, &proof.account)?;
    let words = slots
        .iter()
        .zip(&proof.slots)
        .map(|(slot, nodes)| read_storage_word(account.storage_root, slot, nodes))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((account, words))
}

/// The slots [`read_game_facts`] reads of each contract, in the order of a
/// [`GameProof`]'s slot proofs, the game's being the fast lane's three
/// ([`FAST_GAME_SLOTS`]): what a relay asks `eth_getProof` for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameProofSlots {
    pub anchor_state_registry: Vec<Root>,
    pub dispute_game_factory: Vec<Root>,
    pub game: Vec<Root>,
    pub system_config: Vec<Root>,
    pub superchain_config: Vec<Root>,
}

/// The slots of [`GameProofSlots`] for the game `game` of immutable arguments
/// `args`, `optimism_portal` being `SystemConfig.optimismPortal()` (the pause
/// identifier); `None` when `args` are not of the pinned implementations'
/// length.
pub fn game_proof_slots(
    params: &BaseParams,
    game: &Address,
    args: &[u8],
    optimism_portal: &Address,
) -> Option<GameProofSlots> {
    if args.len() != ARGS_FIXED_BYTES + 32 * params.intermediate_roots {
        return None;
    }
    let root_claim: Root = args[0x14..0x34].try_into().expect("32 bytes");
    let uuid = game_uuid(params.game_type, &root_claim, &args[0x54..]);
    Some(GameProofSlots {
        anchor_state_registry: vec![
            u256(ASR_SYSTEM_CONFIG_SLOT),
            u256(ASR_FACTORY_SLOT),
            mapping_slot(&padded(game), ASR_BLACKLIST_MAPPING),
            u256(ASR_RETIREMENT_SLOT),
            IMPLEMENTATION_SLOT,
        ],
        dispute_game_factory: vec![mapping_slot(&uuid, FACTORY_GAMES_MAPPING), IMPLEMENTATION_SLOT],
        game: FAST_GAME_SLOTS.iter().map(|slot| u256(*slot)).collect(),
        system_config: vec![
            u256(SYSTEM_CONFIG_SUPERCHAIN_SLOT),
            OPTIMISM_PORTAL_SLOT,
            IMPLEMENTATION_SLOT,
        ],
        superchain_config: vec![
            mapping_slot(&[0u8; 32], SUPERCHAIN_PAUSE_MAPPING),
            mapping_slot(&padded(optimism_portal), SUPERCHAIN_PAUSE_MAPPING),
            IMPLEMENTATION_SLOT,
        ],
    })
}

/// Read, under the L1 state root `l1_state_root`, a game and what judges it.
/// Refuses only what is not a proof (or not a game of a pinned
/// implementation); [`judge_game`] decides validity.
pub fn read_game_facts(
    l1_state_root: Root,
    params: &BaseParams,
    proof: &GameProof,
) -> Result<GameFacts, BaseGameError> {
    let args = &proof.args;
    if args.len() != ARGS_FIXED_BYTES + 32 * params.intermediate_roots
        || (proof.game_account.slots.len() != 1
            && proof.game_account.slots.len() != FAST_GAME_SLOTS.len())
    {
        return Err(BaseGameError::ProofShape);
    }
    // The game: its code is a clone of a pinned implementation over exactly
    // these arguments.
    let game = read_account(l1_state_root, &proof.game, &proof.game_account.account)?;
    let game_impl = *params
        .game_impls
        .iter()
        .find(|implementation| keccak256(&clone_code(implementation, args)) == game.code_hash)
        .ok_or(BaseGameError::UnknownGameCode)?;
    let state = read_storage_word(
        game.storage_root,
        &u256(GAME_STATE_SLOT),
        &proof.game_account.slots[0],
    )?;
    // The fast lane's words, when the proof carries them (slots 5 and 7).
    let fast = if proof.game_account.slots.len() == FAST_GAME_SLOTS.len() {
        let countered_by = read_storage_word(
            game.storage_root,
            &u256(GAME_COUNTERED_SLOT),
            &proof.game_account.slots[1],
        )?;
        let word7 = read_storage_word(
            game.storage_root,
            &u256(GAME_PROOF_COUNT_SLOT),
            &proof.game_account.slots[2],
        )?;
        Some(FastGameFacts {
            proof_count: field(&word7, 8, 1)[0],
            countered_by,
        })
    } else {
        None
    };
    let root_claim: Root = args[0x14..0x34].try_into().expect("32 bytes");
    let l2_word: Root = args[0x54..0x74].try_into().expect("32 bytes");
    if l2_word[..24] != [0u8; 24] {
        return Err(BaseGameError::L2BlockOverflow);
    }
    // The factory: the GameId registered for the game's UUID.
    let uuid = game_uuid(params.game_type, &root_claim, &args[0x54..]);
    let (_, factory) = read_contract(
        l1_state_root,
        &params.dispute_game_factory,
        &proof.dispute_game_factory,
        &[
            mapping_slot(&uuid, FACTORY_GAMES_MAPPING),
            IMPLEMENTATION_SLOT,
        ],
    )?;
    // The registry.
    let (_, asr) = read_contract(
        l1_state_root,
        &params.anchor_state_registry,
        &proof.anchor_state_registry,
        &[
            u256(ASR_SYSTEM_CONFIG_SLOT),
            u256(ASR_FACTORY_SLOT),
            mapping_slot(&padded(&proof.game), ASR_BLACKLIST_MAPPING),
            u256(ASR_RETIREMENT_SLOT),
            IMPLEMENTATION_SLOT,
        ],
    )?;
    // SystemConfig: the superchain config and the portal it pauses under.
    let (_, system) = read_contract(
        l1_state_root,
        &params.system_config,
        &proof.system_config,
        &[
            u256(SYSTEM_CONFIG_SUPERCHAIN_SLOT),
            OPTIMISM_PORTAL_SLOT,
            IMPLEMENTATION_SLOT,
        ],
    )?;
    let optimism_portal = field_address(&system[1], 0);
    let (_, superchain) = read_contract(
        l1_state_root,
        &params.superchain_config,
        &proof.superchain_config,
        &[
            mapping_slot(&[0u8; 32], SUPERCHAIN_PAUSE_MAPPING),
            mapping_slot(&padded(&optimism_portal), SUPERCHAIN_PAUSE_MAPPING),
            IMPLEMENTATION_SLOT,
        ],
    )?;
    Ok(GameFacts {
        game: proof.game,
        game_impl,
        root_claim,
        l2_block_number: field_u64(&l2_word, 0, 8),
        created_at: field_u64(&state, 0, 8),
        resolved_at: field_u64(&state, 8, 8),
        status: field(&state, 16, 1)[0],
        initialized: field(&state, 17, 1)[0] != 0,
        was_respected: field(&state, 18, 1)[0] != 0,
        fast,
        // GameId: type (bits 224..256), timestamp (160..224), proxy (0..160).
        registered_proxy: field_address(&factory[0], 0),
        blacklisted: asr[ASR_SLOT_BLACKLIST] != [0u8; 32],
        retirement_timestamp: field_u64(&asr[ASR_SLOT_RETIREMENT], 4, 8),
        asr_system_config: field_address(&asr[ASR_SLOT_SYSTEM_CONFIG], 2),
        asr_dispute_game_factory: field_address(&asr[ASR_SLOT_FACTORY], 0),
        superchain_config: field_address(&system[0], 0),
        optimism_portal,
        pause_global: superchain[0],
        pause_portal: superchain[1],
        implementations: [
            field_address(&asr[ASR_SLOT_IMPLEMENTATION], 0),
            field_address(&factory[1], 0),
            field_address(&system[2], 0),
            field_address(&superchain[2], 0),
        ],
    })
}

/// `SuperchainConfig.paused(id)` at `now`: a pause timestamp set and younger
/// than its expiry. A timestamp beyond 64 bits pauses (or reverts) for ever.
fn pause_active(timestamp: &Root, now: u64, expiry: u64) -> bool {
    if *timestamp == [0u8; 32] {
        return false;
    }
    if timestamp[..24] != [0u8; 24] {
        return true;
    }
    u128::from(now) < u128::from(field_u64(timestamp, 0, 8)) + u128::from(expiry)
}

/// The equivalent of `AnchorStateRegistry.isGameClaimValid(game)` at the L1
/// time `l1_timestamp`, on facts the proofs showed. Pins and wiring first
/// (they give the facts their meaning), then the registry's order.
pub fn judge_game(
    facts: &GameFacts,
    l1_timestamp: u64,
    params: &BaseParams,
) -> Result<ValidGame, BaseGameError> {
    judge_game_proper(facts, l1_timestamp, params)?;
    // isGameFinalized: resolved, then strictly past the delay (a time before
    // the resolution would revert there).
    let resolved = facts.resolved_at != 0
        && (facts.status == GAME_STATUS_CHALLENGER_WINS
            || facts.status == GAME_STATUS_DEFENDER_WINS);
    if !resolved {
        return Err(BaseGameError::NotResolved);
    }
    match l1_timestamp.checked_sub(facts.resolved_at) {
        Some(elapsed) if elapsed > params.finality_delay_seconds => {}
        _ => return Err(BaseGameError::NotFinalized),
    }
    if facts.status != GAME_STATUS_DEFENDER_WINS {
        return Err(BaseGameError::NotDefenderWins);
    }
    Ok(ValidGame {
        game: facts.game,
        l2_block_number: facts.l2_block_number,
        root_claim: facts.root_claim,
    })
}

/// Steps 0-5 of [`judge_game`], the slow return's and the fast lane's alike:
/// pins and wiring, then `isGameProper` and `isGameRespected`.
fn judge_game_proper(
    facts: &GameFacts,
    l1_timestamp: u64,
    params: &BaseParams,
) -> Result<(), BaseGameError> {
    let pinned = [
        (
            params.anchor_state_registry_impl,
            BaseContract::AnchorStateRegistry,
        ),
        (
            params.dispute_game_factory_impl,
            BaseContract::DisputeGameFactory,
        ),
        (params.system_config_impl, BaseContract::SystemConfig),
        (
            params.superchain_config_impl,
            BaseContract::SuperchainConfig,
        ),
    ];
    for (read, (pin, contract)) in facts.implementations.iter().zip(pinned) {
        if *read != pin {
            return Err(BaseGameError::UnknownImplementation(contract));
        }
    }
    if !params.game_impls.contains(&facts.game_impl) {
        return Err(BaseGameError::UnknownGameCode);
    }
    if facts.asr_system_config != params.system_config {
        return Err(BaseGameError::Wiring(BaseContract::SystemConfig));
    }
    if facts.asr_dispute_game_factory != params.dispute_game_factory {
        return Err(BaseGameError::Wiring(BaseContract::DisputeGameFactory));
    }
    if facts.superchain_config != params.superchain_config {
        return Err(BaseGameError::Wiring(BaseContract::SuperchainConfig));
    }
    // isGameProper. (`game.anchorStateRegistry() == this` is an immutable of
    // the pinned implementation.)
    if facts.registered_proxy != facts.game {
        return Err(BaseGameError::NotRegistered);
    }
    if facts.blacklisted {
        return Err(BaseGameError::Blacklisted);
    }
    if facts.created_at <= facts.retirement_timestamp {
        return Err(BaseGameError::Retired);
    }
    if pause_active(
        &facts.pause_global,
        l1_timestamp,
        params.pause_expiry_seconds,
    ) || pause_active(
        &facts.pause_portal,
        l1_timestamp,
        params.pause_expiry_seconds,
    ) {
        return Err(BaseGameError::Paused);
    }
    // isGameRespected.
    if !facts.was_respected {
        return Err(BaseGameError::NotRespected);
    }
    Ok(())
}

/// The fast lane's judgment of a game (D127, RETOUR-RAPIDE SPEC §3): the
/// steps 0-5 of [`judge_game`] unchanged (pinned, wired, registered by the
/// factory, not blacklisted, not retired, not paused, respected), then, in
/// place of finality: in progress, initialized, at least one proof verified
/// (a game is created only with one: `initializeWithInitData` verifies it),
/// not countered.
pub fn judge_game_fast(
    facts: &GameFacts,
    l1_timestamp: u64,
    params: &BaseParams,
) -> Result<ValidGame, BaseGameError> {
    judge_game_proper(facts, l1_timestamp, params)?;
    if facts.status != GAME_STATUS_IN_PROGRESS {
        return Err(BaseGameError::NotInProgress);
    }
    if !facts.initialized {
        return Err(BaseGameError::NotInitialized);
    }
    let fast = facts.fast.ok_or(BaseGameError::FastFactsMissing)?;
    if fast.proof_count == 0 {
        return Err(BaseGameError::NoProof);
    }
    if fast.countered_by != [0u8; 32] {
        return Err(BaseGameError::Countered);
    }
    Ok(ValidGame {
        game: facts.game,
        l2_block_number: facts.l2_block_number,
        root_claim: facts.root_claim,
    })
}

/// A game valid at the finalized L1 block of state root `l1_state_root` and
/// time `l1_timestamp`.
pub fn verify_valid_game(
    l1_state_root: Root,
    l1_timestamp: u64,
    params: &BaseParams,
    proof: &GameProof,
) -> Result<ValidGame, BaseGameError> {
    judge_game(
        &read_game_facts(l1_state_root, params, proof)?,
        l1_timestamp,
        params,
    )
}

/// The output root of `preimage`.
pub fn output_root(preimage: &OutputRootPreimage) -> Root {
    let mut bytes = [0u8; 128];
    bytes[..32].copy_from_slice(&preimage.version);
    bytes[32..64].copy_from_slice(&preimage.state_root);
    bytes[64..96].copy_from_slice(&preimage.message_passer_storage_root);
    bytes[96..].copy_from_slice(&preimage.block_hash);
    keccak256(&bytes)
}

/// The Base state root a root claim commits to: the preimage must be of
/// version 0 and hash to the claim.
pub fn open_output_root(
    root_claim: &Root,
    preimage: &OutputRootPreimage,
) -> Result<Root, BaseReturnError> {
    if preimage.version != [0u8; 32] {
        return Err(BaseReturnError::OutputVersion);
    }
    if output_root(preimage) != *root_claim {
        return Err(BaseReturnError::OutputRootMismatch);
    }
    Ok(preimage.state_root)
}

/// A Base state root Base holds final at the L1 block read: a valid game and
/// the preimage of its claim.
pub fn verify_base_state_root(
    l1_state_root: Root,
    l1_timestamp: u64,
    params: &BaseParams,
    proof: &GameProof,
    preimage: &OutputRootPreimage,
) -> Result<BaseState, BaseReturnError> {
    let game = verify_valid_game(l1_state_root, l1_timestamp, params, proof)?;
    Ok(BaseState {
        game: game.game,
        l2_block_number: game.l2_block_number,
        state_root: open_output_root(&game.root_claim, preimage)?,
    })
}

/// The whole return read: a Base state root Base holds final, then, under it,
/// the bridge's code and genesis and `withdrawals[n] = keccak(owner ‖ amount)`.
pub fn verify_base_release(
    l1_state_root: Root,
    l1_timestamp: u64,
    params: &BaseParams,
    proof: &GameProof,
    preimage: &OutputRootPreimage,
    read: &BridgeRead<'_>,
) -> Result<BaseState, BaseReturnError> {
    let base = verify_base_state_root(l1_state_root, l1_timestamp, params, proof, preimage)?;
    verify_bridge_account(
        base.state_root,
        read.bridge,
        read.account_proof,
        read.genesis_proof,
    )
    .map_err(BaseReturnError::Bridge)?;
    verify_withdrawal(
        base.state_root,
        read.bridge,
        read.account_proof,
        read.n,
        read.owner,
        read.amount,
        read.storage_proof,
    )
    .map_err(BaseReturnError::Bridge)?;
    Ok(base)
}

// ---------------------------------------------------------------------------
// The section objects of the return over Base (D107: nodes deduplicated)
// ---------------------------------------------------------------------------

/// Most distinct nodes one `BaseGameProof` object carries (64-67 measured).
pub const MAX_GAME_PROOF_NODES: usize = 256;
/// Longest argument list a game proof carries (776 bytes for type 621).
pub const MAX_GAME_ARGS_BYTES: usize = 2_048;
/// Most slot proofs of one contract (the rule reads at most five).
pub const MAX_CONTRACT_SLOTS: usize = 8;
/// Bytes of an `OutputRootPreimage` object: version, state root, message
/// passer storage root, block hash.
pub const OUTPUT_ROOT_PREIMAGE_BYTES: usize = 128;

/// Why an object's payload is not a Base game proof or an output preimage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaseCodecError {
    Truncated,
    TrailingBytes,
    TooManyNodes,
    NodeTooLong,
    TooManySlots,
    ArgsTooLong,
    /// A node index outside the node table.
    BadIndex,
    /// Not the canonical encoding: a node twice in the table, a node no
    /// proof references, or the table out of first-reference order.
    NonCanonical,
}

/// The `BaseGameProof` object payload (D107, nodes deduplicated): the game,
/// its arguments, the table of the distinct proof nodes in the order the
/// proofs first reference them, then the five contracts (registry, factory,
/// game, `SystemConfig`, `SuperchainConfig`), each its account proof and its
/// slot proofs as lists of indices into the table.
///
/// ```text
/// payload  = game:20 ‖ args_len:u16 ‖ args ‖ nodes:u16 ‖ (len:u16 ‖ node)* ‖ contract{5}
/// contract = list ‖ slots:u8 ‖ list{slots}
/// list     = count:u8 ‖ index:u16{count}
/// ```
///
/// Integers little-endian. One encoding per proof: the decoder refuses any
/// other ([`BaseCodecError::NonCanonical`]).
pub fn encode_game_proof(proof: &GameProof) -> Vec<u8> {
    let contracts = game_proof_contracts(proof);
    let mut table: Vec<&[u8]> = Vec::new();
    let mut index: std::collections::HashMap<&[u8], u16> = std::collections::HashMap::new();
    let mut lists: Vec<Vec<u16>> = Vec::new();
    for contract in contracts {
        for list in std::iter::once(&contract.account).chain(contract.slots.iter()) {
            let mut indices = Vec::with_capacity(list.len());
            for node in list {
                let at = match index.get(node.as_slice()) {
                    Some(at) => *at,
                    None => {
                        let at = u16::try_from(table.len()).expect("node table fits u16");
                        table.push(node);
                        index.insert(node, at);
                        at
                    }
                };
                indices.push(at);
            }
            lists.push(indices);
        }
    }
    let mut out = proof.game.to_vec();
    out.extend_from_slice(
        &u16::try_from(proof.args.len())
            .expect("arguments fit u16")
            .to_le_bytes(),
    );
    out.extend_from_slice(&proof.args);
    out.extend_from_slice(
        &u16::try_from(table.len())
            .expect("table fits u16")
            .to_le_bytes(),
    );
    for node in &table {
        out.extend_from_slice(
            &u16::try_from(node.len())
                .expect("node fits u16")
                .to_le_bytes(),
        );
        out.extend_from_slice(node);
    }
    let mut lists = lists.into_iter();
    let mut put = |out: &mut Vec<u8>| {
        let indices = lists.next().expect("one list per proof");
        out.push(u8::try_from(indices.len()).expect("list fits u8"));
        for at in indices {
            out.extend_from_slice(&at.to_le_bytes());
        }
    };
    for contract in contracts {
        put(&mut out);
        out.push(u8::try_from(contract.slots.len()).expect("slots fit u8"));
        for _ in &contract.slots {
            put(&mut out);
        }
    }
    out
}

/// The five contracts of a game proof, in the order of its object.
fn game_proof_contracts(proof: &GameProof) -> [&ContractProof; 5] {
    [
        &proof.anchor_state_registry,
        &proof.dispute_game_factory,
        &proof.game_account,
        &proof.system_config,
        &proof.superchain_config,
    ]
}

fn take_bytes<'a>(src: &mut &'a [u8], n: usize) -> Result<&'a [u8], BaseCodecError> {
    if src.len() < n {
        return Err(BaseCodecError::Truncated);
    }
    let (head, tail) = src.split_at(n);
    *src = tail;
    Ok(head)
}

fn take_u16(src: &mut &[u8]) -> Result<usize, BaseCodecError> {
    Ok(usize::from(u16::from_le_bytes(
        take_bytes(src, 2)?.try_into().expect("two bytes"),
    )))
}

fn take_u8(src: &mut &[u8]) -> Result<usize, BaseCodecError> {
    Ok(usize::from(take_bytes(src, 1)?[0]))
}

/// Decode [`encode_game_proof`]; every bound is checked before allocating,
/// and only the canonical encoding is accepted.
pub fn decode_game_proof(bytes: &[u8]) -> Result<GameProof, BaseCodecError> {
    use crate::consensus::vault::{MAX_PROOF_NODES, MAX_PROOF_NODE_BYTES};
    let mut src = bytes;
    let game: Address = take_bytes(&mut src, 20)?.try_into().expect("20 bytes");
    let args_len = take_u16(&mut src)?;
    if args_len > MAX_GAME_ARGS_BYTES {
        return Err(BaseCodecError::ArgsTooLong);
    }
    let args = take_bytes(&mut src, args_len)?.to_vec();
    let count = take_u16(&mut src)?;
    if count > MAX_GAME_PROOF_NODES {
        return Err(BaseCodecError::TooManyNodes);
    }
    let mut table: Vec<&[u8]> = Vec::with_capacity(count);
    for _ in 0..count {
        let len = take_u16(&mut src)?;
        if len == 0 || len > MAX_PROOF_NODE_BYTES {
            return Err(BaseCodecError::NodeTooLong);
        }
        table.push(take_bytes(&mut src, len)?);
    }
    let list = |src: &mut &[u8]| -> Result<Nodes, BaseCodecError> {
        let len = take_u8(src)?;
        if len > MAX_PROOF_NODES {
            return Err(BaseCodecError::TooManyNodes);
        }
        (0..len)
            .map(|_| {
                let at = take_u16(src)?;
                table
                    .get(at)
                    .map(|node| node.to_vec())
                    .ok_or(BaseCodecError::BadIndex)
            })
            .collect()
    };
    let contract = |src: &mut &[u8]| -> Result<ContractProof, BaseCodecError> {
        let account = list(src)?;
        let slots = take_u8(src)?;
        if slots > MAX_CONTRACT_SLOTS {
            return Err(BaseCodecError::TooManySlots);
        }
        Ok(ContractProof {
            account,
            slots: (0..slots).map(|_| list(src)).collect::<Result<_, _>>()?,
        })
    };
    let proof = GameProof {
        game,
        args,
        anchor_state_registry: contract(&mut src)?,
        dispute_game_factory: contract(&mut src)?,
        game_account: contract(&mut src)?,
        system_config: contract(&mut src)?,
        superchain_config: contract(&mut src)?,
    };
    if !src.is_empty() {
        return Err(BaseCodecError::TrailingBytes);
    }
    // One encoding per proof: each node once, each referenced, in the order
    // of first reference.
    if encode_game_proof(&proof) != bytes {
        return Err(BaseCodecError::NonCanonical);
    }
    Ok(proof)
}

/// The `OutputRootPreimage` object payload: its four words, in order.
pub fn encode_output_root_preimage(preimage: &OutputRootPreimage) -> Vec<u8> {
    [
        preimage.version,
        preimage.state_root,
        preimage.message_passer_storage_root,
        preimage.block_hash,
    ]
    .concat()
}

/// Decode [`encode_output_root_preimage`]: exactly 128 bytes.
pub fn decode_output_root_preimage(bytes: &[u8]) -> Result<OutputRootPreimage, BaseCodecError> {
    match bytes.len().cmp(&OUTPUT_ROOT_PREIMAGE_BYTES) {
        std::cmp::Ordering::Less => return Err(BaseCodecError::Truncated),
        std::cmp::Ordering::Greater => return Err(BaseCodecError::TrailingBytes),
        std::cmp::Ordering::Equal => {}
    }
    let word = |i: usize| -> Root { bytes[32 * i..32 * (i + 1)].try_into().expect("32 bytes") };
    Ok(OutputRootPreimage {
        version: word(0),
        state_root: word(1),
        message_passer_storage_root: word(2),
        block_hash: word(3),
    })
}

#[cfg(test)]
pub(crate) mod tests;
