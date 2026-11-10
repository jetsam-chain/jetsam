// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The fast lane of the return over Base, relay side (D127, D130,
//! RETOUR-RAPIDE SPEC §10): which game of Base the relay signs.
//!
//! The relay signs `(game, L2 block, root claim)` with its fast-lane key only
//! after checking the proposal itself:
//!
//! * the game was created in a **finalized** L1 block (it is among the games
//!   the factory counts at the finalized block), is of the pinned type and
//!   the factory's implementation for it is a pinned one;
//! * it is in progress, proven at least once, not countered, proper and
//!   respected in the eyes of Base's registry (at the finalized block);
//! * its L2 block is at or below the **finalized** head of every Base RPC it
//!   reads (the L2 data is on the L1 and final);
//! * its root claim is recomputed, `keccak(0³² ‖ stateRoot ‖ withdrawalsRoot
//!   ‖ hash)`, from the header of that block as **at least two independent
//!   Base RPCs** serve it, all equal;
//! * no ancestor it knows (up the `parentAddress` chain to an ancestor Base
//!   resolved `DEFENDER_WINS`) was won by a challenger or blacklisted.
//!
//! Otherwise it signs nothing and says why (`ALERT FAST …`). The decision is
//! a pure function ([`judge`]) of what was read ([`GameView`],
//! [`BaseHeaderView`], [`AncestorView`]); the reading only uses `eth_call`
//! and `eth_getBlockByNumber` ([`read_game`], [`read_base`]). A dry run
//! ([`dry_run`]) signs with a throwaway key and sends nothing.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context};
use blst::min_pk::SecretKey;
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::base_read::{
    BaseParams, GAME_STATUS_CHALLENGER_WINS, GAME_STATUS_DEFENDER_WINS, GAME_STATUS_IN_PROGRESS,
};
use jetsam_chain::consensus::destination::fast::{
    verify_attestation, FastAttestation, FastStatement, FAST_RELEASE_DST, KEY_BYTES,
};
use jetsam_chain::consensus::destination::mpt::keccak256;

use crate::base_capture::{hex20, hex32_of, BaseRpc};

#[cfg(test)]
#[path = "fast_tests.rs"]
pub(crate) mod tests;

/// Fewest independent Base RPCs a root is recomputed from.
pub const MIN_BASE_RPCS: usize = 2;
/// Deepest the relay walks up a game's ancestors before it gives up (no
/// signature): 5 days of proposals (≈ 360) and a margin.
pub const MAX_ANCESTOR_DEPTH: usize = 1_000;

/// What the relay reads of one game on the L1, at the finalized block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameView {
    pub index: u64,
    /// The factory's `gameCount()` at the finalized block.
    pub count_at_finalized: u64,
    pub game_type: u32,
    pub game: [u8; 20],
    /// The factory's implementation of the type at the finalized block.
    pub factory_impl: [u8; 20],
    pub status: u8,
    pub proof_count: u8,
    pub countered_by: [u8; 32],
    pub l2_block: u64,
    pub root_claim: [u8; 32],
    pub parent: [u8; 20],
    /// The registry's `isGameProper` and `isGameRespected`.
    pub proper: bool,
    pub respected: bool,
}

/// What one Base RPC serves: its finalized head, and the header of the
/// game's L2 block (`None`: not served).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BaseHeaderView {
    pub finalized: u64,
    pub header: Option<HeaderWords>,
}

/// The words of a Base header the output root reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeaderWords {
    pub number: u64,
    pub state_root: [u8; 32],
    pub withdrawals_root: [u8; 32],
    pub hash: [u8; 32],
}

impl HeaderWords {
    /// The output root of version 0.
    pub fn output_root(&self) -> [u8; 32] {
        let mut preimage = [0u8; 128];
        preimage[32..64].copy_from_slice(&self.state_root);
        preimage[64..96].copy_from_slice(&self.withdrawals_root);
        preimage[96..].copy_from_slice(&self.hash);
        keccak256(&preimage)
    }
}

/// One ancestor of a game, as the relay reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AncestorView {
    pub game: [u8; 20],
    pub status: u8,
    pub blacklisted: bool,
}

/// Why the relay does not sign.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The game is not among those the factory counts at the finalized L1
    /// block.
    NotCreatedInAFinalizedBlock {
        index: u64,
        count: u64,
    },
    /// Another game type than the pinned one.
    OtherType(u32),
    /// The factory's implementation is not a pinned one: Base upgraded.
    UnknownImplementation([u8; 20]),
    NotInProgress(u8),
    NoProof,
    Countered,
    /// Not proper or not respected in the eyes of Base's registry.
    NotProper,
    NotRespected,
    /// Fewer Base RPCs answered than [`MIN_BASE_RPCS`].
    TooFewBaseRpcs(usize),
    /// The L2 block is above a Base RPC's finalized head.
    NotFinalizedOnBase {
        l2_block: u64,
        finalized: u64,
    },
    /// A Base RPC did not serve the header.
    HeaderNotServed,
    /// Two Base RPCs serve different headers for the block.
    BaseRpcsDisagree,
    /// The root recomputed is not the game's claim.
    RootMismatch {
        claimed: [u8; 32],
        computed: [u8; 32],
    },
    /// An ancestor was won by a challenger.
    AncestorChallenged([u8; 20]),
    AncestorBlacklisted([u8; 20]),
    /// The ancestors were not walked up to one resolved `DEFENDER_WINS`.
    AncestorsUnknown,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// What the statement names besides the game: this relay's network, chains
/// and bridge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatementDomain {
    pub network: [u8; 32],
    pub l1_chain_id: u64,
    pub base_chain_id: u64,
    pub bridge: [u8; 20],
}

/// The relay's decision on one game: the statement to sign, or why not.
pub fn judge(
    game: &GameView,
    params: &BaseParams,
    bases: &[BaseHeaderView],
    ancestors: &[AncestorView],
    domain: &StatementDomain,
) -> Result<FastStatement, Refusal> {
    // On the L1, at the finalized block.
    if game.index >= game.count_at_finalized {
        return Err(Refusal::NotCreatedInAFinalizedBlock {
            index: game.index,
            count: game.count_at_finalized,
        });
    }
    if game.game_type != params.game_type {
        return Err(Refusal::OtherType(game.game_type));
    }
    if !params.game_impls.contains(&game.factory_impl) {
        return Err(Refusal::UnknownImplementation(game.factory_impl));
    }
    if game.status != GAME_STATUS_IN_PROGRESS {
        return Err(Refusal::NotInProgress(game.status));
    }
    if game.proof_count == 0 {
        return Err(Refusal::NoProof);
    }
    if game.countered_by != [0u8; 32] {
        return Err(Refusal::Countered);
    }
    if !game.proper {
        return Err(Refusal::NotProper);
    }
    if !game.respected {
        return Err(Refusal::NotRespected);
    }
    // On Base: two RPCs at least, the block final on each, the same header
    // from each, and its output root the claim.
    if bases.len() < MIN_BASE_RPCS {
        return Err(Refusal::TooFewBaseRpcs(bases.len()));
    }
    let mut served = Vec::with_capacity(bases.len());
    for base in bases {
        if base.finalized < game.l2_block {
            return Err(Refusal::NotFinalizedOnBase {
                l2_block: game.l2_block,
                finalized: base.finalized,
            });
        }
        match base.header {
            Some(header) if header.number == game.l2_block => served.push(header),
            _ => return Err(Refusal::HeaderNotServed),
        }
    }
    if served.iter().any(|h| *h != served[0]) {
        return Err(Refusal::BaseRpcsDisagree);
    }
    let computed = served[0].output_root();
    if computed != game.root_claim {
        return Err(Refusal::RootMismatch {
            claimed: game.root_claim,
            computed,
        });
    }
    // Its ancestors, up to one Base resolved for the defender.
    let mut grounded = false;
    for ancestor in ancestors {
        if ancestor.status == GAME_STATUS_CHALLENGER_WINS {
            return Err(Refusal::AncestorChallenged(ancestor.game));
        }
        if ancestor.blacklisted {
            return Err(Refusal::AncestorBlacklisted(ancestor.game));
        }
        if ancestor.status == GAME_STATUS_DEFENDER_WINS {
            grounded = true;
            break;
        }
    }
    if !grounded {
        return Err(Refusal::AncestorsUnknown);
    }
    Ok(FastStatement {
        network: domain.network,
        l1_chain_id: domain.l1_chain_id,
        base_chain_id: domain.base_chain_id,
        bridge: domain.bridge,
        game: game.game,
        l2_block: game.l2_block,
        root_claim: game.root_claim,
    })
}

/// The relay's attestation of `statement` under its fast-lane key `secret`.
pub fn sign(statement: &FastStatement, secret: &SecretKey) -> FastAttestation {
    FastAttestation {
        statement: *statement,
        signature: secret
            .sign(&statement.message(), FAST_RELEASE_DST, &[])
            .compress(),
    }
}

/// A throwaway key for a dry run: derived from `label` (never written
/// anywhere); its public key, to check the signatures.
pub fn throwaway_key(label: &str) -> anyhow::Result<(SecretKey, [u8; KEY_BYTES])> {
    let ikm = keccak256(format!("jetsam-fast-dry-run/THROWAWAY/{label}").as_bytes());
    let secret =
        SecretKey::key_gen(&ikm, &[]).map_err(|e| anyhow!("no key from the label: {e:?}"))?;
    let public = secret.sk_to_pk().compress();
    Ok((secret, public))
}

// ---------------------------------------------------------------------------
// The keys (fast-keygen): the relay's key and the cold revocation key
// ---------------------------------------------------------------------------

/// First line of a secret file written by [`fast_keygen`].
pub const FAST_SECRET_FILE_TAG: &str = "jetsam-fast-bls-secret-v1";

/// A new BLS12-381 min-pk secret from the system's randomness, written to
/// `out` (created, never over an existing file, mode 0600), never printed;
/// its public key (what a profile's `FAST_RELEASE_KEY` or `FAST_REVOKE_KEY`
/// holds).
pub fn fast_keygen(out: &Path) -> anyhow::Result<[u8; KEY_BYTES]> {
    use std::io::Write as _;
    use std::os::unix::fs::OpenOptionsExt;
    let mut ikm = zeroize::Zeroizing::new([0u8; 32]);
    getrandom::getrandom(&mut *ikm).map_err(|e| anyhow!("no randomness from the system: {e}"))?;
    let secret =
        SecretKey::key_gen(&*ikm, &[]).map_err(|e| anyhow!("no key from the randomness: {e:?}"))?;
    let bytes = zeroize::Zeroizing::new(secret.to_bytes());
    let text = zeroize::Zeroizing::new(format!(
        "{FAST_SECRET_FILE_TAG}\n{}\n",
        hex::encode(&*bytes)
    ));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(out)
        .with_context(|| format!("{}: not created (an existing file is never replaced)", out.display()))?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    Ok(secret.sk_to_pk().compress())
}

/// The secret of a file [`fast_keygen`] wrote; refused when the file is
/// readable or writable by anyone but its owner.
pub fn load_fast_secret(path: &Path) -> anyhow::Result<SecretKey> {
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(path)
        .with_context(|| format!("{}", path.display()))?
        .permissions()
        .mode();
    if mode & 0o077 != 0 {
        bail!(
            "{}: mode {:o}, readable or writable by others than its owner: refused",
            path.display(),
            mode & 0o777
        );
    }
    let text = zeroize::Zeroizing::new(std::fs::read_to_string(path)?);
    let mut lines = text.lines();
    if lines.next() != Some(FAST_SECRET_FILE_TAG) {
        bail!("{}: not a fast-lane secret file", path.display());
    }
    let bytes = zeroize::Zeroizing::new(
        hex::decode(lines.next().unwrap_or_default().trim())
            .map_err(|_| anyhow!("{}: not a fast-lane secret file", path.display()))?,
    );
    if bytes.len() != 32 || lines.next().is_some() {
        bail!("{}: not a fast-lane secret file", path.display());
    }
    SecretKey::from_bytes(&bytes).map_err(|_| anyhow!("{}: not a BLS secret", path.display()))
}

// ---------------------------------------------------------------------------
// Reading (eth_call, eth_getBlockByNumber only)
// ---------------------------------------------------------------------------

fn selector(signature: &str) -> [u8; 4] {
    keccak256(signature.as_bytes())[..4]
        .try_into()
        .expect("4 bytes")
}

fn word_u64(x: u64) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[24..].copy_from_slice(&x.to_be_bytes());
    w
}

fn word_address(a: &[u8; 20]) -> [u8; 32] {
    let mut w = [0u8; 32];
    w[12..].copy_from_slice(a);
    w
}

/// `eth_call` of `signature(args)` at `to`, at the L1 block `block` (hex
/// quantity or tag).
pub(crate) fn call(
    rpc: &dyn BaseRpc,
    to: &[u8; 20],
    signature: &str,
    args: &[[u8; 32]],
    block: &str,
) -> anyhow::Result<Vec<u8>> {
    let mut data = selector(signature).to_vec();
    for a in args {
        data.extend_from_slice(a);
    }
    let v = rpc.rpc(
        "eth_call",
        json!([{"to": format!("0x{}", hex::encode(to)), "data": format!("0x{}", hex::encode(&data))}, block]),
    )?;
    let text = v
        .as_str()
        .ok_or_else(|| anyhow!("eth_call {signature}: not bytes"))?;
    Ok(hex::decode(text.trim_start_matches("0x"))?)
}

fn word_at(bytes: &[u8], i: usize) -> anyhow::Result<[u8; 32]> {
    bytes
        .get(32 * i..32 * (i + 1))
        .map(|w| w.try_into().expect("32"))
        .ok_or_else(|| anyhow!("answer too short"))
}

fn u64_of(word: &[u8; 32]) -> anyhow::Result<u64> {
    if word[..24] != [0u8; 24] {
        bail!("a word beyond 64 bits");
    }
    Ok(u64::from_be_bytes(word[24..].try_into().expect("8")))
}

fn address_of(word: &[u8; 32]) -> [u8; 20] {
    word[12..].try_into().expect("20")
}

/// The L1 block the relay reads at: the finalized one (its number).
pub fn finalized_l1_block(l1: &dyn BaseRpc) -> anyhow::Result<u64> {
    let b = l1.rpc("eth_getBlockByNumber", json!(["finalized", false]))?;
    crate::l1::hex_u64(&b["number"])
}

/// The factory's `gameCount()` at the L1 block `block`.
pub fn game_count(l1: &dyn BaseRpc, params: &BaseParams, block: u64) -> anyhow::Result<u64> {
    let tag = format!("0x{block:x}");
    u64_of(&word_at(
        &call(l1, &params.dispute_game_factory, "gameCount()", &[], &tag)?,
        0,
    )?)
}

/// Game `index` of the factory as the L1 holds it at block `block`.
pub fn read_game(
    l1: &dyn BaseRpc,
    params: &BaseParams,
    index: u64,
    block: u64,
) -> anyhow::Result<GameView> {
    let tag = format!("0x{block:x}");
    let factory = &params.dispute_game_factory;
    let count = game_count(l1, params, block)?;
    if index >= count {
        // Not created by the finalized block: nothing else to read.
        return Ok(GameView {
            index,
            count_at_finalized: count,
            game_type: 0,
            game: [0; 20],
            factory_impl: [0; 20],
            status: 0,
            proof_count: 0,
            countered_by: [0; 32],
            l2_block: 0,
            root_claim: [0; 32],
            parent: [0; 20],
            proper: false,
            respected: false,
        });
    }
    let at = call(
        l1,
        factory,
        "gameAtIndex(uint256)",
        &[word_u64(index)],
        &tag,
    )?;
    let game_type = u64_of(&word_at(&at, 0)?)? as u32;
    let game = address_of(&word_at(&at, 2)?);
    let factory_impl = address_of(&word_at(
        &call(
            l1,
            factory,
            "gameImpls(uint32)",
            &[word_u64(u64::from(game_type))],
            &tag,
        )?,
        0,
    )?);
    let get = |signature: &str| -> anyhow::Result<[u8; 32]> {
        word_at(&call(l1, &game, signature, &[], &tag)?, 0)
    };
    let asr = |signature: &str| -> anyhow::Result<bool> {
        Ok(u64_of(&word_at(
            &call(
                l1,
                &params.anchor_state_registry,
                signature,
                &[word_address(&game)],
                &tag,
            )?,
            0,
        )?)? != 0)
    };
    Ok(GameView {
        index,
        count_at_finalized: count,
        game_type,
        game,
        factory_impl,
        status: u64_of(&get("status()")?)? as u8,
        proof_count: u64_of(&get("proofCount()")?)? as u8,
        countered_by: get("counteredByIntermediateRootIndexPlusOne()")?,
        l2_block: u64_of(&get("l2SequenceNumber()")?)?,
        root_claim: get("rootClaim()")?,
        parent: address_of(&get("parentAddress()")?),
        proper: asr("isGameProper(address)")?,
        respected: asr("isGameRespected(address)")?,
    })
}

/// One ancestor at the L1 block `tag` (the latest: a challenger's win and a
/// blacklisting are never undone, so the newest reading is the safest): its
/// status, whether the registry blacklisted it, and its parent.
pub fn read_ancestor(
    l1: &dyn BaseRpc,
    params: &BaseParams,
    game: &[u8; 20],
    tag: &str,
) -> anyhow::Result<(AncestorView, [u8; 20])> {
    let status = u64_of(&word_at(&call(l1, game, "status()", &[], tag)?, 0)?)? as u8;
    let parent = address_of(&word_at(&call(l1, game, "parentAddress()", &[], tag)?, 0)?);
    let blacklisted = u64_of(&word_at(
        &call(
            l1,
            &params.anchor_state_registry,
            "isGameBlacklisted(address)",
            &[word_address(game)],
            tag,
        )?,
        0,
    )?)? != 0;
    Ok((
        AncestorView {
            game: *game,
            status,
            blacklisted,
        },
        parent,
    ))
}

/// Ancestors read in one JSON-RPC batch (two calls each: `status()` and
/// `isGameBlacklisted`; a third, `parentAddress()`, for the one whose
/// parent is not known yet).
pub const ANCESTOR_BATCH: usize = 50;

/// The parents of the games the relay walked. A game's `parentAddress()` is
/// set when it is created and never changes (an immutable argument of the
/// clone, in none of its storage slots: measured on game #30 081 of Base
/// Sepolia), so it is read once and kept, across passes, failed passes and
/// restarts ([`load_parents`], [`store_parents`]). NOTHING else of an
/// ancestor is kept: its status and its blacklisting can change, and every
/// walk reads them again ([`read_ancestors`]).
pub type ParentCache = BTreeMap<[u8; 20], [u8; 20]>;

/// What one walk did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkStats {
    /// Ancestors read (their status and blacklisting, in this walk).
    pub ancestors: usize,
    /// Requests sent (a batch is one).
    pub requests: usize,
    /// `eth_call`s asked.
    pub calls: usize,
    /// Parents read (not known before the walk).
    pub parents_read: usize,
    /// Batches the endpoint refused, asked again one call at a time.
    pub batches_refused: usize,
}

/// `calls` (`eth_call` objects at the L1 block `tag`) in one batch, or one by
/// one when the endpoint refuses the batch; their answers' bytes, in order.
pub fn call_many(
    rpc: &dyn BaseRpc,
    calls: &[([u8; 20], Vec<u8>)],
    tag: &str,
    stats: &mut WalkStats,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let params: Vec<Value> = calls
        .iter()
        .map(|(to, data)| {
            json!([{"to": format!("0x{}", hex::encode(to)), "data": format!("0x{}", hex::encode(data))}, tag])
        })
        .collect();
    let batch: Vec<(&str, Value)> = params.iter().map(|p| ("eth_call", p.clone())).collect();
    stats.calls += calls.len();
    stats.requests += 1;
    let answers = match rpc.rpc_batch(&batch) {
        Ok(answers) if answers.len() == calls.len() => answers,
        _ => {
            stats.batches_refused += 1;
            let mut answers = Vec::with_capacity(calls.len());
            for (method, p) in batch {
                stats.requests += 1;
                answers.push(rpc.rpc(method, p)?);
            }
            answers
        }
    };
    answers
        .iter()
        .map(|v| {
            let text = v.as_str().ok_or_else(|| anyhow!("eth_call: not bytes"))?;
            Ok(hex::decode(text.trim_start_matches("0x"))?)
        })
        .collect()
}

/// The ancestors of a game, from its `parent` up to the first one resolved
/// (`DEFENDER_WINS` or `CHALLENGER_WINS`, included), at most
/// [`MAX_ANCESTOR_DEPTH`], read at the L1 block `tag`.
///
/// **Freshness**: the status and the blacklisting of every ancestor returned
/// are read in THIS walk (never kept from an earlier one); only the parents
/// are taken from `parents` (and the ones read added to it). They are read
/// [`ANCESTOR_BATCH`] at a time along the parents known (a stretch may read
/// a few games past the one the walk stops at: harmless), one at a time,
/// with its parent, where no parent is known. `progress` is told after each
/// ancestor read.
pub fn read_ancestors(
    l1: &dyn BaseRpc,
    params: &BaseParams,
    parent: [u8; 20],
    tag: &str,
    parents: &mut ParentCache,
    progress: &mut dyn FnMut(&WalkStats, &ParentCache),
) -> anyhow::Result<(Vec<AncestorView>, WalkStats)> {
    let mut stats = WalkStats::default();
    let mut out = Vec::new();
    let mut at = parent;
    while at != [0u8; 20] && out.len() < MAX_ANCESTOR_DEPTH {
        // The next stretch: along the parents known, at most a batch.
        let mut stretch = vec![at];
        while stretch.len() < ANCESTOR_BATCH && out.len() + stretch.len() < MAX_ANCESTOR_DEPTH {
            match parents.get(stretch.last().expect("one")) {
                Some(p) if *p != [0u8; 20] => stretch.push(*p),
                _ => break,
            }
        }
        let last = *stretch.last().expect("one");
        // A lone game whose parent is not known (a walk where none is): its parent in the same request.
        let with_parent = stretch.len() == 1 && !parents.contains_key(&last);
        let mut calls = Vec::with_capacity(2 * stretch.len() + 1);
        for game in &stretch {
            calls.push((*game, selector("status()").to_vec()));
            let mut data = selector("isGameBlacklisted(address)").to_vec();
            data.extend_from_slice(&word_address(game));
            calls.push((params.anchor_state_registry, data));
        }
        if with_parent {
            calls.push((last, selector("parentAddress()").to_vec()));
        }
        let answers = call_many(l1, &calls, tag, &mut stats)?;
        if with_parent {
            parents.insert(last, address_of(&word_at(&answers[2 * stretch.len()], 0)?));
            stats.parents_read += 1;
        }
        for (i, game) in stretch.iter().enumerate() {
            let view = AncestorView {
                game: *game,
                status: u64_of(&word_at(&answers[2 * i], 0)?)? as u8,
                blacklisted: u64_of(&word_at(&answers[2 * i + 1], 0)?)? != 0,
            };
            out.push(view);
            stats.ancestors += 1;
            progress(&stats, parents);
            if view.status == GAME_STATUS_DEFENDER_WINS || view.status == GAME_STATUS_CHALLENGER_WINS {
                return Ok((out, stats));
            }
        }
        // The walk goes on past the stretch: the parent of its last game (read now if not known).
        at = match parents.get(&last) {
            Some(p) => *p,
            None => {
                let answer = call_many(l1, &[(last, selector("parentAddress()").to_vec())], tag, &mut stats)?;
                let p = address_of(&word_at(&answer[0], 0)?);
                parents.insert(last, p);
                stats.parents_read += 1;
                p
            }
        };
    }
    Ok((out, stats))
}

/// A walk says how far it is every this many ancestors read.
pub const WALK_LOG_EVERY: usize = 50;

/// [`read_ancestors`] (at the latest L1 block) with its journal: a line when
/// the walk of game #`index` begins, one every [`WALK_LOG_EVERY`] ancestors
/// read, one when it ends (how many, how long, how many requests); `keep` is
/// handed the parents at each of those lines and at the end, failed or not
/// (what was read is never lost).
pub fn walk_ancestors_logged(
    l1: &dyn BaseRpc,
    params: &BaseParams,
    index: u64,
    parent: [u8; 20],
    parents: &mut ParentCache,
    say: &dyn Fn(&str),
    keep: &mut dyn FnMut(&ParentCache),
) -> anyhow::Result<Vec<AncestorView>> {
    let started = std::time::Instant::now();
    say(&format!(
        "game #{index}: walking its ancestors from 0x{} ({} parents known)",
        hex::encode(parent),
        parents.len()
    ));
    let result = read_ancestors(l1, params, parent, "latest", parents, &mut |stats, parents| {
        if stats.ancestors % WALK_LOG_EVERY == 0 {
            say(&format!(
                "game #{index}: {} ancestors read in {:.1} s ({} requests)",
                stats.ancestors,
                started.elapsed().as_secs_f64(),
                stats.requests
            ));
            keep(parents);
        }
    });
    keep(parents);
    let secs = started.elapsed().as_secs_f64();
    match &result {
        Ok((views, stats)) => say(&format!(
            "game #{index}: {} ancestors walked in {secs:.1} s ({} requests, {} calls, {} parents read, {} batches refused), the last {}",
            views.len(),
            stats.requests,
            stats.calls,
            stats.parents_read,
            stats.batches_refused,
            views
                .last()
                .map(|v| format!("0x{} status {} blacklisted {}", hex::encode(v.game), v.status, v.blacklisted))
                .unwrap_or_else(|| "none".into())
        )),
        Err(e) => say(&format!("game #{index}: the walk of its ancestors failed after {secs:.1} s: {e:#}")),
    }
    result.map(|(views, _)| views)
}

/// The parents kept in `path` (none when there is no file yet).
pub fn load_parents(path: &Path) -> anyhow::Result<ParentCache> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ParentCache::new()),
        Err(e) => return Err(e.into()),
    };
    let v: Value = serde_json::from_str(&text).with_context(|| format!("{}", path.display()))?;
    let mut out = ParentCache::new();
    for pair in v["parents"]
        .as_array()
        .ok_or_else(|| anyhow!("{}: no parents", path.display()))?
    {
        out.insert(
            hex20(pair[0].as_str().unwrap_or_default())?,
            hex20(pair[1].as_str().unwrap_or_default())?,
        );
    }
    Ok(out)
}

/// Keep `parents` in `path` (written then renamed).
pub fn store_parents(path: &Path, parents: &ParentCache) -> anyhow::Result<()> {
    let v = json!({
        "what": "game -> parentAddress() (never changes); nothing else of an ancestor is kept",
        "parents": parents
            .iter()
            .map(|(g, p)| json!([format!("0x{}", hex::encode(g)), format!("0x{}", hex::encode(p))]))
            .collect::<Vec<_>>(),
    });
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec(&v)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// What one Base RPC serves for `l2_block`.
pub fn read_base(rpc: &dyn BaseRpc, l2_block: u64) -> anyhow::Result<BaseHeaderView> {
    let fin = rpc.rpc("eth_getBlockByNumber", json!(["finalized", false]))?;
    let finalized = crate::l1::hex_u64(&fin["number"])?;
    let h = rpc.rpc(
        "eth_getBlockByNumber",
        json!([format!("0x{l2_block:x}"), false]),
    )?;
    let header = if h.is_null() {
        None
    } else {
        let word = |key: &str| -> anyhow::Result<[u8; 32]> {
            hex32_of(h[key].as_str().ok_or_else(|| anyhow!("header: no {key}"))?)
        };
        Some(HeaderWords {
            number: crate::l1::hex_u64(&h["number"])?,
            state_root: word("stateRoot")?,
            withdrawals_root: word("withdrawalsRoot")?,
            hash: word("hash")?,
        })
    };
    Ok(BaseHeaderView { finalized, header })
}

// ---------------------------------------------------------------------------
// The dry run: a throwaway key, nothing sent
// ---------------------------------------------------------------------------

/// One line of the dry run's journal.
pub fn journal_line(
    index: u64,
    game: &GameView,
    verdict: &Result<FastAttestation, Refusal>,
) -> Value {
    match verdict {
        Ok(a) => json!({
            "index": index,
            "game": format!("0x{}", hex::encode(game.game)),
            "l2_block": game.l2_block,
            "root_claim": format!("0x{}", hex::encode(game.root_claim)),
            "signed": true,
            "attestation": format!("0x{}", hex::encode(a.encode_payload())),
        }),
        Err(why) => json!({
            "index": index,
            "game": format!("0x{}", hex::encode(game.game)),
            "l2_block": game.l2_block,
            "signed": false,
            "refusal": why.to_string(),
        }),
    }
}

/// `jg-relay fast-dry-run …`: the fast lane's checks on real games of Base,
/// signed with a throwaway key, **nothing sent** (RPC reads only).
pub fn dry_run_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || {
        anyhow!(
            "fast-dry-run --l1 <url> --base <url> --base <url> [--base <url>…] --bridge <0x…20> --journal <file> \
             [--games <n newest, 3>] [--key-label <label>] [--watch-seconds <s>] [--capture-dir <dir> --proposal-anchor <L2 block>] \
             [--parents <file: the ancestors' parents, kept>]"
        )
    };
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let values = |flag: &str| -> Vec<String> {
        args.iter()
            .enumerate()
            .filter(|(_, a)| *a == flag)
            .filter_map(|(i, _)| args.get(i + 1).cloned())
            .collect()
    };
    let l1 = crate::l1::L1::new(&value("--l1").ok_or_else(usage)?);
    let bases: Vec<crate::l1::L1> = values("--base")
        .iter()
        .map(|u| crate::l1::L1::new(u))
        .collect();
    let bridge = hex20(&value("--bridge").ok_or_else(usage)?)?;
    let journal = std::path::PathBuf::from(value("--journal").ok_or_else(usage)?);
    let games: u64 = value("--games")
        .map(|g| g.parse())
        .transpose()?
        .unwrap_or(3);
    let label = value("--key-label").unwrap_or_else(|| format!("{}", crate::now_unix()));
    let watch: u64 = value("--watch-seconds")
        .map(|w| w.parse())
        .transpose()?
        .unwrap_or(0);
    let capture_dir = value("--capture-dir").map(std::path::PathBuf::from);
    let anchor: Option<u64> = value("--proposal-anchor").map(|a| a.parse()).transpose()?;
    let params = jetsam_chain::consensus::destination::state::BASE_RETURN_PARAMS;
    let rules = jetsam_chain::consensus::destination::fast::FastLaneRules::CONSENSUS;
    let domain = StatementDomain {
        network: jetsam_chain::consensus::guardian::network_id(),
        l1_chain_id: rules.l1_chain_id,
        base_chain_id: rules.base_chain_id,
        bridge,
    };
    let (secret, public) = throwaway_key(&label)?;
    let say = |line: &str| println!("{} jg-relay fast-dry-run {line}", crate::now_unix());
    say(&format!(
        "THROWAWAY key 0x{} (label {label}): nothing is sent; L1 chain {}, Base chain {}, {} Base RPCs, bridge 0x{}",
        hex::encode(public),
        domain.l1_chain_id,
        domain.base_chain_id,
        bases.len(),
        hex::encode(bridge)
    ));
    // The parents of the ancestors, kept in --parents <file> when given (a later run reads none of them again).
    let parents_file = value("--parents").map(std::path::PathBuf::from);
    let mut parents = match &parents_file {
        Some(path) => load_parents(path)?,
        None => ParentCache::new(),
    };
    let mut seen = std::collections::BTreeSet::new();
    let started = crate::now_unix();
    let mut first = true;
    loop {
        // A read that fails (a public RPC reset, a pruned state) is an ALERT and is tried again at the next pass;
        // it never leads to a signature (every check reads first).
        if let (Some(dir), Some(anchor), Some(base)) = (&capture_dir, anchor, bases.first()) {
            let pass = || -> anyhow::Result<crate::base_capture::PassReport> {
                let head = crate::l1::hex_u64(&base.rpc("eth_blockNumber", json!([]))?)?;
                let plan = crate::base_capture::CapturePlan {
                    address: bridge,
                    slots: crate::base_capture::capture_slots(13, 9, &[]),
                    anchor,
                    interval: crate::base_capture::PROPOSAL_INTERVAL_BLOCKS,
                    window: crate::base_capture::CAPTURE_WINDOW_BLOCKS,
                };
                crate::base_capture::capture_pass(base, &plan, head, crate::now_unix(), dir)
            };
            match pass() {
                Ok(report) => {
                    for b in &report.captured {
                        say(&format!("captured the bridge at Base block {b}"));
                    }
                    for (b, why) in &report.refused {
                        say(&format!("ALERT FAST CAPTURE {b}: {why}"));
                    }
                }
                Err(e) => say(&format!("ALERT FAST CAPTURE pass: {e:#}")),
            }
        }
        let count = finalized_l1_block(&l1).and_then(|fin| game_count(&l1, &params, fin));
        let count = match count {
            Ok(count) => count,
            Err(e) => {
                say(&format!("ALERT FAST RPC the game count: {e:#}"));
                0
            }
        };
        let from = if first {
            count.saturating_sub(games)
        } else {
            count.saturating_sub(games.max(4))
        };
        if count > 0 {
            first = false;
        }
        for index in from..count {
            if !seen.insert(index) {
                continue;
            }
            let mut one = || -> anyhow::Result<()> {
                // Read at a finalized block taken now (a public RPC keeps little history).
                let fin = finalized_l1_block(&l1)?;
                let game = read_game(&l1, &params, index, fin)?;
                let mut views = Vec::with_capacity(bases.len());
                for base in &bases {
                    match read_base(base, game.l2_block) {
                        Ok(view) => views.push(view),
                        Err(e) => say(&format!("ALERT FAST BASE_RPC {}: {e:#}", base.url)),
                    }
                }
                parents.insert(game.game, game.parent);
                let ancestors = walk_ancestors_logged(&l1, &params, index, game.parent, &mut parents, &say, &mut |p| {
                    if let Some(path) = &parents_file {
                        if let Err(e) = store_parents(path, p) {
                            say(&format!("ALERT FAST the parents not kept in {}: {e:#}", path.display()));
                        }
                    }
                })?;
                let verdict =
                    judge(&game, &params, &views, &ancestors, &domain).map(|s| sign(&s, &secret));
                let mut line = journal_line(index, &game, &verdict);
                line["l1_finalized"] = json!(fin);
                line["ancestors_walked"] = json!(ancestors.len());
                if let Some(dir) = &capture_dir {
                    line["capture"] =
                        json!(crate::base_capture::load(dir, game.l2_block)?.is_some());
                }
                // The control: the same game with its claim altered is never signed.
                let mut altered = game;
                altered.root_claim[31] ^= 1;
                let control = judge(&altered, &params, &views, &ancestors, &domain);
                line["control_altered_root_refused"] = json!(control.is_err());
                match &verdict {
                    Ok(_) => say(&format!(
                        "SIGNED game #{index} 0x{} L2 {} root 0x{} (control refused: {})",
                        hex::encode(game.game),
                        game.l2_block,
                        hex::encode(game.root_claim),
                        control.is_err()
                    )),
                    Err(why) => say(&format!("ALERT FAST game #{index} not signed: {why}")),
                }
                if control.is_ok() {
                    bail!("game #{index}: an altered root would have been signed");
                }
                use std::io::Write as _;
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&journal)?;
                writeln!(file, "{line}")?;
                Ok(())
            };
            if let Err(e) = one() {
                if e.to_string().contains("an altered root would have been signed") {
                    return Err(e);
                }
                say(&format!("ALERT FAST RPC game #{index}: {e:#} (tried again at the next pass)"));
                seen.remove(&index);
            }
        }
        if crate::now_unix() >= started + watch {
            break;
        }
        std::thread::sleep(std::time::Duration::from_secs(60));
    }
    let signed = check_journal(&journal, &public, 0)?;
    say(&format!(
        "journal {}: {signed} signed attestations verify under the throwaway key",
        journal.display()
    ));
    Ok(())
}

/// Read and check `path`'s JSON lines (the journal) for at least `want`
/// signed games: every attestation verifies under `public`.
pub fn check_journal(path: &Path, public: &[u8; KEY_BYTES], want: usize) -> anyhow::Result<usize> {
    let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
    let mut signed = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        if v["signed"].as_bool() == Some(true) {
            let bytes = hex::decode(
                v["attestation"]
                    .as_str()
                    .unwrap_or_default()
                    .trim_start_matches("0x"),
            )?;
            let a = FastAttestation::decode_payload(&bytes).map_err(|e| anyhow!("{e:?}"))?;
            verify_attestation(&a, Some(public)).map_err(|e| anyhow!("{e:?}"))?;
            if hex20(v["game"].as_str().unwrap_or_default())? != a.statement.game {
                bail!("a journal line names another game than its attestation");
            }
            signed += 1;
        }
    }
    if signed < want {
        bail!("{signed} signed games, {want} wanted");
    }
    Ok(signed)
}
