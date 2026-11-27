// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The slow lane of the return over Base, relay side (RET24 SUP §6.3).
//!
//! A burn on Base the fast lane did not pay (its relay key closed by a
//! contradiction or a revocation, no key armed in this build, or simply not
//! done) is released under the newest dispute game Base holds valid
//! (`AnchorStateRegistry.isGameClaimValid`, judged exactly as consensus
//! judges it, `base_read::judge_game`) whose Base block is at or after the
//! burn's: one game frees every burn before it. The intent is
//! `ReleaseIntent { base: Some(..), fast: None }` (kind 3), built with the
//! consensus's own encoders and checked, before anything leaves, by the
//! functions `verify_releases_on_base` runs (all of them but the finality
//! update's BLS signature, which only the node's committee keys verify).
//!
//! Rules (each one a test of `slow_release_tests.rs`):
//! * never a burn the branch paid (`getAdvances.released`), by either lane;
//!   the one exception is the one consensus makes: a fast release
//!   contradicted and not yet repaid to its real burner (`getFastLane`);
//! * never a burn whose fast release this relay sent less than
//!   `fast_grace_secs` ago while its relay key is open on the branch;
//! * the release group pays what V2 asks at the node's next block
//!   (`getAdvances.payee`: the creditor of the advance `JTMADV` of exactly
//!   that burn, else its owner) and carries the slow release's marker;
//! * at most what one block carries: the releases of one pass share one
//!   finality update, game proof, preimage and account proof, and fit the
//!   section's 48 KiB with no committee update beside them (D107 §9.1),
//!   `max_releases_per_block` and the V6 window's room; each spends vault
//!   slots no other intent of this relay holds;
//! * a write-ahead journal (`slow_release.json`, written whole and synced
//!   before the node is called): an intent handed in is never handed in
//!   again before `resubmit_secs`, nor more than `max_attempts` times, and a
//!   restart resumes from it;
//! * a dry run (the default) builds, checks and writes each intent once and
//!   sends nothing; only `send` hands intents to the node.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{anyhow, bail, Context};
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::advance::advance_marker_kind;
use jetsam_chain::consensus::destination::base_read::{
    decode_game_proof, decode_output_root_preimage, encode_game_proof, encode_output_root_preimage,
    open_output_root, verify_base_state_root, BaseParams, GameProof, OutputRootPreimage, ValidGame,
    GAME_STATUS_DEFENDER_WINS,
};
use jetsam_chain::consensus::destination::l1_read::{
    genesis_slot, verify_bridge_account, verify_withdrawal,
};
use jetsam_chain::consensus::destination::light_client::{finalized_execution, FinalizedExecution};
use jetsam_chain::consensus::destination::ssz::LightClientFinalityUpdate;
use jetsam_chain::consensus::destination::state::DestinationRules;
use jetsam_chain::consensus::destination::{
    destination_section_wire_len, DestinationObject, DestinationObjectKind, DestinationSection,
    MAX_BLOCK_DESTINATION_OBJECTS,
};
use jetsam_chain::consensus::vault::{
    check_release_group_marked, decode_bridge_account_proof, encode_bridge_account_proof,
    VaultRelease, VaultRules,
};
use jetsam_chain::consensus::wire_limits::MAX_DESTINATION_SECTION_BYTES;
use jetsam_mempool::destination::{BaseReleaseProofs, DestinationIntent, ReleaseIntent};
use jetsam_poseidon2b::primitives::Address;
use jetsam_tx::{PagedSpendIntent, TxPage};

use crate::base_capture::Capture;
use crate::fast::CapturedGame;
use crate::fast_release::{bridge_under, pack_group, select_vault_slots, AssemblyRefusal, Burn, VaultSlot};
use crate::fast_run::{slow_game_under, BurnSeen};

#[cfg(test)]
#[path = "slow_release_tests.rs"]
mod tests;

/// What the slow lane does with what it builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlowMode {
    /// No slow release is looked at.
    Off,
    /// Built, checked and written, once per burn; nothing is sent.
    DryRun,
    /// Handed to the node (only on explicit configuration).
    Send,
}

impl Default for SlowMode {
    fn default() -> Self {
        Self::DryRun
    }
}

impl SlowMode {
    /// `off`, `dry-run` or `send`.
    pub fn parse(text: &str) -> anyhow::Result<Self> {
        match text {
            "off" => Ok(Self::Off),
            "dry-run" => Ok(Self::DryRun),
            "send" => Ok(Self::Send),
            other => bail!("--slow-release {other:?}: off, dry-run or send"),
        }
    }
}

/// A fast release this relay sent may still land this long (its relay key
/// open on the branch): the slow lane leaves its burn alone meanwhile.
pub const FAST_GRACE_SECS: u64 = 6 * 3_600;
/// An intent handed to the node and not paid after this long is built
/// again (fresh finality update, game, vault slots).
pub const RESUBMIT_SECS: u64 = 2 * 3_600;
/// At most this many intents handed in per burn; then the operator decides.
pub const MAX_ATTEMPTS: u32 = 6;

/// The slow lane's settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlowConfig {
    pub mode: SlowMode,
    pub fast_grace_secs: u64,
    pub resubmit_secs: u64,
    pub max_attempts: u32,
}

impl Default for SlowConfig {
    fn default() -> Self {
        Self {
            mode: SlowMode::DryRun,
            fast_grace_secs: FAST_GRACE_SECS,
            resubmit_secs: RESUBMIT_SECS,
            max_attempts: MAX_ATTEMPTS,
        }
    }
}

// ---------------------------------------------------------------------------
// The journal
// ---------------------------------------------------------------------------

/// Where a burn stands for the slow lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SlowStatus {
    /// Built by a dry run under the game of Base block `l2_block`, written;
    /// never sent.
    DryRun { at: u64, l2_block: u64, txid: String },
    /// Written before it was handed to the node (write-ahead), under the
    /// game of Base block `l2_block`; `admitted` once the node said so. Its
    /// vault slots are held from this relay's other intents.
    Submitted {
        at: u64,
        l2_block: u64,
        txid: String,
        admitted: bool,
        slots: Vec<u32>,
    },
    /// The branch paid `n` (either lane).
    Paid { at: u64 },
    /// Handed in `max_attempts` times and never paid: the operator decides.
    GaveUp { at: u64 },
}

/// One burn the slow lane looked at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlowEntry {
    pub owner: Address,
    pub amount: u64,
    /// Intents handed to the node so far.
    pub attempts: u32,
    pub status: SlowStatus,
}

/// What the slow lane remembers across restarts, by withdrawal index `n`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlowJournal {
    pub entries: BTreeMap<u64, SlowEntry>,
}

fn status_json(status: &SlowStatus) -> Value {
    match status {
        SlowStatus::DryRun { at, l2_block, txid } => {
            json!({"status": "dry_run", "at": at, "l2_block": l2_block, "txid": txid})
        }
        SlowStatus::Submitted {
            at,
            l2_block,
            txid,
            admitted,
            slots,
        } => json!({
            "status": "submitted", "at": at, "l2_block": l2_block, "txid": txid,
            "admitted": admitted, "slots": slots,
        }),
        SlowStatus::Paid { at } => json!({"status": "paid", "at": at}),
        SlowStatus::GaveUp { at } => json!({"status": "gave_up", "at": at}),
    }
}

fn status_of(v: &Value) -> anyhow::Result<SlowStatus> {
    let number = |key: &str| v[key].as_u64().ok_or_else(|| anyhow!("no {key}"));
    let text = |key: &str| {
        v[key]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| anyhow!("no {key}"))
    };
    Ok(match v["status"].as_str() {
        Some("dry_run") => SlowStatus::DryRun {
            at: number("at")?,
            l2_block: number("l2_block")?,
            txid: text("txid")?,
        },
        Some("submitted") => SlowStatus::Submitted {
            at: number("at")?,
            l2_block: number("l2_block")?,
            txid: text("txid")?,
            admitted: v["admitted"].as_bool().ok_or_else(|| anyhow!("no admitted"))?,
            slots: v["slots"]
                .as_array()
                .ok_or_else(|| anyhow!("no slots"))?
                .iter()
                .map(|slot| {
                    slot.as_u64()
                        .and_then(|slot| u32::try_from(slot).ok())
                        .ok_or_else(|| anyhow!("a slot that is no slot index"))
                })
                .collect::<anyhow::Result<_>>()?,
        },
        Some("paid") => SlowStatus::Paid { at: number("at")? },
        Some("gave_up") => SlowStatus::GaveUp { at: number("at")? },
        other => bail!("status {other:?}"),
    })
}

impl SlowJournal {
    /// The journal kept in `path` (empty when there is none yet); an
    /// unreadable one is an error, never an empty journal.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(anyhow!("{}: {e}", path.display())),
        };
        let v: Value = serde_json::from_str(&text)
            .with_context(|| format!("{}: not the slow lane's journal", path.display()))?;
        let mut entries = BTreeMap::new();
        for e in v["entries"]
            .as_array()
            .ok_or_else(|| anyhow!("{}: no entries", path.display()))?
        {
            let n = e["n"]
                .as_u64()
                .ok_or_else(|| anyhow!("{}: an entry without n", path.display()))?;
            let read = || -> anyhow::Result<SlowEntry> {
                Ok(SlowEntry {
                    owner: Address(crate::base_capture::hex32_of(
                        e["owner"].as_str().ok_or_else(|| anyhow!("no owner"))?,
                    )?),
                    amount: e["amount"].as_u64().ok_or_else(|| anyhow!("no amount"))?,
                    attempts: u32::try_from(
                        e["attempts"].as_u64().ok_or_else(|| anyhow!("no attempts"))?,
                    )?,
                    status: status_of(e)?,
                })
            };
            let entry = read().with_context(|| format!("{}: entry {n}", path.display()))?;
            if entries.insert(n, entry).is_some() {
                bail!("{}: entry {n} twice", path.display());
            }
        }
        Ok(Self { entries })
    }

    /// Written whole and synced ([`crate::write_atomic`]).
    pub fn store(&self, path: &Path) -> anyhow::Result<()> {
        let entries: Vec<Value> = self
            .entries
            .iter()
            .map(|(n, e)| {
                let mut v = status_json(&e.status);
                v["n"] = json!(n);
                v["owner"] = json!(format!("0x{}", hex::encode(e.owner.0)));
                v["amount"] = json!(e.amount);
                v["attempts"] = json!(e.attempts);
                v
            })
            .collect();
        let v = json!({
            "what": "the slow lane of the return over Base: each burn looked at, by withdrawal index",
            "entries": entries,
        });
        crate::write_atomic(path, &serde_json::to_vec_pretty(&v)?)
    }

    /// Its entry of `burn` (of exactly that owner and amount).
    fn entry_of(&self, burn: &Burn) -> Option<&SlowEntry> {
        self.entries
            .get(&burn.n)
            .filter(|e| e.owner == burn.owner && e.amount == burn.amount)
    }
}

// ---------------------------------------------------------------------------
// What the branch says of a burn
// ---------------------------------------------------------------------------

/// `getAdvances(n, owner, amount)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchBurn {
    /// The return over Base governs the node's next block.
    pub active: bool,
    /// The branch paid `n` (D52).
    pub released: bool,
    /// Whom V2 asks the release's first output to pay at the next block.
    pub payee: Address,
}

/// `getAdvances`' answer.
pub fn branch_burn_of(v: &Value) -> anyhow::Result<BranchBurn> {
    let payee = v["payee"]
        .as_str()
        .ok_or_else(|| anyhow!("getAdvances: no payee"))?;
    Ok(BranchBurn {
        active: v["active"].as_bool() == Some(true),
        released: v["released"]
            .as_bool()
            .ok_or_else(|| anyhow!("getAdvances: no released"))?,
        payee: Address::parse(payee).map_err(|e| anyhow!("getAdvances: payee {e:?}"))?,
    })
}

/// One fast release the branch keeps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FastRecordView {
    pub owner: Address,
    pub amount: u64,
    pub contradicted: bool,
    /// Its real burner was paid by a slow release since.
    pub consumed: bool,
}

/// `getFastLane`: whether the branch closed this build's relay key, and the
/// fast releases it keeps.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FastLaneView {
    pub key_closed: bool,
    pub records: BTreeMap<u64, FastRecordView>,
}

/// `getFastLane`'s answer.
pub fn fast_lane_view_of(v: &Value) -> anyhow::Result<FastLaneView> {
    let mut records = BTreeMap::new();
    for r in v["records"].as_array().map(Vec::as_slice).unwrap_or_default() {
        let n = r["n"]
            .as_u64()
            .ok_or_else(|| anyhow!("getFastLane: a record without n"))?;
        records.insert(
            n,
            FastRecordView {
                owner: Address(crate::base_capture::hex32_of(
                    r["owner"].as_str().unwrap_or_default(),
                )?),
                amount: r["amount"]
                    .as_u64()
                    .ok_or_else(|| anyhow!("getFastLane: a record without amount"))?,
                contradicted: r["contradicted"].as_bool().unwrap_or(false),
                consumed: r["consumed"].as_bool().unwrap_or(false),
            },
        );
    }
    Ok(FastLaneView {
        key_closed: v["release_key_closed"].as_bool().unwrap_or(false),
        records,
    })
}

/// What the slow lane does with one burn this pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The branch paid it (either lane): nothing more, ever.
    Paid,
    /// Its fast release, sent by this relay at `since`, may still land.
    FastPending { since: u64 },
    /// Handed to the node at `since`: it waits for its block.
    InFlight { since: u64 },
    /// A dry run built it already.
    DryRunDone,
    /// Handed in `max_attempts` times and never paid.
    GaveUp,
    /// Its owner is an advance marker: never payable over Base.
    NeverPayable,
    /// The slow lane releases it.
    Release,
}

/// The slow lane's decision for `burn`: the branch's word on it, the fast
/// lane's, when this relay sent its fast release (`fast_sent`), the
/// journal's entry, at `now`.
pub fn judge_burn(
    burn: &Burn,
    branch: &BranchBurn,
    lane: &FastLaneView,
    fast_sent: Option<u64>,
    entry: Option<&SlowEntry>,
    cfg: &SlowConfig,
    now: u64,
) -> Verdict {
    // A fast release of `n` the branch contradicted, its real burner not
    // paid yet: consensus lets one slow release pay another burn than it.
    let reopened = lane.records.get(&burn.n).is_some_and(|r| {
        r.contradicted && !r.consumed && (r.owner, r.amount) != (burn.owner, burn.amount)
    });
    if branch.released && !reopened {
        return Verdict::Paid;
    }
    if advance_marker_kind(&burn.owner).is_some() {
        return Verdict::NeverPayable;
    }
    if let Some(since) = fast_sent {
        if !lane.key_closed && !reopened && now < since.saturating_add(cfg.fast_grace_secs) {
            return Verdict::FastPending { since };
        }
    }
    let entry = entry.filter(|e| e.owner == burn.owner && e.amount == burn.amount);
    match entry.map(|e| (&e.status, e.attempts)) {
        Some((SlowStatus::GaveUp { .. }, _)) => Verdict::GaveUp,
        Some((SlowStatus::DryRun { .. }, _)) if cfg.mode == SlowMode::DryRun => Verdict::DryRunDone,
        Some((SlowStatus::Submitted { at, .. }, attempts)) => {
            if now < at.saturating_add(cfg.resubmit_secs) {
                Verdict::InFlight { since: *at }
            } else if attempts >= cfg.max_attempts {
                Verdict::GaveUp
            } else {
                Verdict::Release
            }
        }
        _ => Verdict::Release,
    }
}

// ---------------------------------------------------------------------------
// The objects, under the newest valid game
// ---------------------------------------------------------------------------

/// The objects of the slow releases of one pass, each checked natively.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlowReleaseObjects {
    /// The L1 block they are read under (the finality update's).
    pub l1: FinalizedExecution,
    pub game: ValidGame,
    /// The payloads they share: the finality update, the game proof
    /// (consensus encoding, nodes deduplicated), the output root preimage,
    /// the bridge's account proof with its `genesisId`.
    pub finality: Vec<u8>,
    pub game_proof: Vec<u8>,
    pub output_preimage: Vec<u8>,
    pub account_proof: Vec<u8>,
    /// One per burn read under the game, by increasing `n`.
    pub releases: Vec<VaultRelease>,
}

/// The slow releases of `burns` under the L1 block `l1`: the game of
/// `proof` judged as consensus judges a slow release (`judge_game`), its
/// claim opened with the capture of its Base block, the bridge's account
/// and `genesisId`, then each burn's `withdrawals[n]`. A refusal of the game
/// or the capture is the pass's; a burn's own refusal is listed with it.
pub fn slow_objects_under(
    l1: &FinalizedExecution,
    finality: &[u8],
    proof: &GameProof,
    capture: &Capture,
    burns: &[Burn],
    rules: &DestinationRules,
) -> Result<(SlowReleaseObjects, Vec<(u64, AssemblyRefusal)>), AssemblyRefusal> {
    let bridge = rules.bridge.ok_or(AssemblyRefusal::BridgeUnset)?;
    // The game: the slow judgment (`isGameClaimValid`), never the fast one.
    let game = slow_game_under(l1, &rules.base_return.base, proof)?;
    // The pass's: the capture of the game's block and of the bridge, the
    // claim opened with it, the bridge's code and genesis under it.
    if capture.block != game.l2_block_number {
        return Err(AssemblyRefusal::CaptureOfAnotherBlock {
            capture: capture.block,
            l2_block: game.l2_block_number,
        });
    }
    if capture.address != bridge.address {
        return Err(AssemblyRefusal::CaptureOfAnotherAccount);
    }
    let genesis = capture
        .storage
        .iter()
        .find(|s| s.key == genesis_slot(bridge.genesis_id_slot))
        .map(|s| s.proof.clone())
        .ok_or(AssemblyRefusal::SlotNotCaptured("genesisId"))?;
    let preimage = OutputRootPreimage {
        version: [0u8; 32],
        state_root: capture.state_root,
        message_passer_storage_root: capture.withdrawals_root,
        block_hash: capture.block_hash,
    };
    let state_root =
        open_output_root(&game.root_claim, &preimage).map_err(AssemblyRefusal::OutputRoot)?;
    verify_bridge_account(state_root, &bridge, &capture.account_proof, &genesis)
        .map_err(AssemblyRefusal::Bridge)?;
    // Each burn's: `withdrawals[n]` under the Base state root.
    let mut ordered = burns.to_vec();
    ordered.sort_by_key(|b| b.n);
    let (mut releases, mut refused) = (Vec::new(), Vec::new());
    for burn in ordered {
        match bridge_under(&game, capture, &burn, &bridge) {
            Ok((_, _, release)) => releases.push(release),
            Err(why) => refused.push((burn.n, why)),
        }
    }
    Ok((
        SlowReleaseObjects {
            l1: l1.clone(),
            game,
            finality: finality.to_vec(),
            game_proof: encode_game_proof(proof),
            output_preimage: encode_output_root_preimage(&preimage),
            account_proof: encode_bridge_account_proof(&capture.account_proof, &genesis),
            releases,
        },
        refused,
    ))
}

/// [`slow_objects_under`] the L1 block the finality update `finality` names.
pub fn assemble_slow(
    finality: &[u8],
    proof: &GameProof,
    capture: &Capture,
    burns: &[Burn],
    rules: &DestinationRules,
) -> Result<(SlowReleaseObjects, Vec<(u64, AssemblyRefusal)>), AssemblyRefusal> {
    let update = LightClientFinalityUpdate::decode_payload(finality)
        .map_err(|_| AssemblyRefusal::FinalityPayload)?;
    let l1 = finalized_execution(&update).map_err(AssemblyRefusal::FinalityExecution)?;
    slow_objects_under(&l1, finality, proof, capture, burns, rules)
}

/// The section objects every release of `objects` shares: the finality
/// update, the game proof, the output root preimage, the account proof.
pub fn shared_objects(objects: &SlowReleaseObjects) -> Vec<DestinationObject> {
    [
        (DestinationObjectKind::FinalityUpdate, &objects.finality),
        (DestinationObjectKind::BaseGameProof, &objects.game_proof),
        (DestinationObjectKind::OutputRootPreimage, &objects.output_preimage),
        (DestinationObjectKind::BridgeAccountProof, &objects.account_proof),
    ]
    .into_iter()
    .map(|(kind, payload)| DestinationObject {
        kind,
        payload: payload.clone(),
    })
    .collect()
}

/// How many of `releases` (in order) one block carries beside `shared`:
/// within the section's object count and 48 KiB (no committee update beside
/// a game proof, D107 §9.1: the whole budget is theirs) and
/// `max_releases`.
pub fn fit_in_one_block(
    shared: &[DestinationObject],
    releases: &[VaultRelease],
    max_releases: usize,
) -> usize {
    let mut section = DestinationSection {
        objects: shared.to_vec(),
    };
    let mut fits = 0;
    for release in releases.iter().take(max_releases) {
        section.objects.push(release.object());
        if section.objects.len() > MAX_BLOCK_DESTINATION_OBJECTS
            || destination_section_wire_len(&section) > MAX_DESTINATION_SECTION_BYTES
        {
            break;
        }
        fits += 1;
    }
    fits
}

/// The release group of a slow release: `payee` receives `amount − R`, then
/// the slow release's marker, then the change to the vault; its fee is `R`
/// (D59). Checked against the block's gate before it is returned, with `R`.
pub fn slow_release_group(
    release: &VaultRelease,
    payee: &Address,
    inputs: &[VaultSlot],
    epoch_anchor: [u8; 32],
    slot_hints: &[u32],
    rules: &VaultRules,
    floor: u64,
) -> anyhow::Result<(Vec<TxPage>, u64)> {
    let total = inputs.iter().map(|s| s.amount).sum::<u64>();
    if total < release.amount {
        bail!(
            "the vault slots hold {total} µ, the release takes {} µ",
            release.amount
        );
    }
    let change = total - release.amount;
    let r = rules.release_fee_for(inputs.len() as u64, 2 + u64::from(change > 0));
    let paid = release
        .amount
        .checked_sub(r)
        .context("a release of less than R")?;
    let marker = release.marker();
    let mut outputs = vec![(*payee, paid), (marker, 0)];
    if change > 0 {
        outputs.push((rules.vault, change));
    }
    let pages = pack_group(rules.vault, inputs, &outputs, r, epoch_anchor, slot_hints)?;
    check_release_group_marked(&pages, release, rules, floor, payee, &marker)
        .map_err(|e| anyhow!("the block's gate refuses the group: {e:?}"))?;
    Ok((pages, r))
}

/// The bytes `submitDestinationObject` takes: the slow release of `release`
/// under `objects`, paid by `payment` (kind 3: Base objects, no attestation).
pub fn slow_release_intent(
    objects: &SlowReleaseObjects,
    release: &VaultRelease,
    payment: PagedSpendIntent,
) -> anyhow::Result<Vec<u8>> {
    DestinationIntent::Release(ReleaseIntent {
        release: release.clone(),
        finality: objects.finality.clone(),
        account_proof: objects.account_proof.clone(),
        payment,
        base: Some(BaseReleaseProofs {
            game_proof: objects.game_proof.clone(),
            output_preimage: objects.output_preimage.clone(),
        }),
        fast: None,
    })
    .encode()
    .map_err(|e| anyhow!("the intent: {e}"))
}

/// The intent `bytes` checked as the node will check it, but the finality
/// update's BLS signature: decoded as the node decodes it (a slow release
/// over Base), the game valid under the update's root and time
/// (`verify_base_state_root`: `judge_game`, then the output root), the
/// bridge's code, genesis and `withdrawals[n]` under the Base state root,
/// then its group against the gate with V2's `payee` and the slow marker.
pub fn verify_slow_intent(
    bytes: &[u8],
    rules: &DestinationRules,
    vault_rules: &VaultRules,
    floor: u64,
    payee: &Address,
) -> anyhow::Result<ReleaseIntent> {
    let decoded = DestinationIntent::decode_with(bytes, true)
        .map_err(|e| anyhow!("a node would not decode it: {e}"))?;
    let DestinationIntent::Release(intent) = decoded else {
        bail!("not a release");
    };
    let base = intent
        .base
        .as_ref()
        .ok_or_else(|| anyhow!("a release without its Base objects"))?;
    if intent.fast.is_some() {
        bail!("a fast release, not a slow one");
    }
    // V7 over Base, as `verify_releases_on_base` reads it (but the BLS).
    let update = LightClientFinalityUpdate::decode_payload(&intent.finality)
        .map_err(|e| anyhow!("the finality update: {e:?}"))?;
    let l1 = finalized_execution(&update).map_err(|e| anyhow!("the finality update: {e:?}"))?;
    let time = l1
        .timestamp
        .ok_or_else(|| anyhow!("the finality update gives no L1 time"))?;
    let game = decode_game_proof(&base.game_proof).map_err(|e| anyhow!("the game proof: {e:?}"))?;
    let preimage = decode_output_root_preimage(&base.output_preimage)
        .map_err(|e| anyhow!("the output root preimage: {e:?}"))?;
    let state = verify_base_state_root(
        l1.state_root,
        time,
        &rules.base_return.base,
        &game,
        &preimage,
    )
    .map_err(|e| anyhow!("the game or its output root: {e:?}"))?;
    let bridge = rules
        .bridge
        .ok_or_else(|| anyhow!("this build has no bridge"))?;
    let (account, genesis) = decode_bridge_account_proof(&intent.account_proof)
        .map_err(|e| anyhow!("the account proof: {e:?}"))?;
    verify_bridge_account(state.state_root, &bridge, &account, &genesis)
        .map_err(|e| anyhow!("the bridge under the Base state root: {e:?}"))?;
    let release = &intent.release;
    verify_withdrawal(
        state.state_root,
        &bridge,
        &account,
        release.n,
        &release.owner.0,
        release.amount,
        &release.storage_proof,
    )
    .map_err(|e| anyhow!("withdrawals[{}] under the Base state root: {e:?}", release.n))?;
    // Its group: V1-V6, V2's payee, the slow release's marker.
    check_release_group_marked(
        &intent.payment.pages,
        release,
        vault_rules,
        floor,
        payee,
        &release.marker(),
    )
    .map_err(|e| anyhow!("the block's gate refuses the group: {e:?}"))?;
    Ok(intent)
}

/// The vault slots this relay's intents in flight hold, but `n`'s own.
pub fn held_slots(journal: &SlowJournal, n: u64) -> BTreeSet<u32> {
    journal
        .entries
        .iter()
        .filter(|(m, _)| **m != n)
        .filter_map(|(_, e)| match &e.status {
            SlowStatus::Submitted { slots, .. } => Some(slots.iter().copied()),
            _ => None,
        })
        .flatten()
        .collect()
}

/// What this relay's intents in flight at `now` take from the V6 window.
fn in_flight_amount(journal: &SlowJournal, cfg: &SlowConfig, now: u64) -> u64 {
    journal
        .entries
        .values()
        .filter(|e| {
            matches!(e.status, SlowStatus::Submitted { at, .. } if now < at.saturating_add(cfg.resubmit_secs))
        })
        .map(|e| e.amount)
        .fold(0u64, u64::saturating_add)
}

// ---------------------------------------------------------------------------
// The node
// ---------------------------------------------------------------------------

/// The vault at the node's tip.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VaultView {
    pub tip: u64,
    pub floor: u64,
    pub slots: Vec<VaultSlot>,
    pub released_in_window: u64,
    pub window_cap: u64,
}

/// What the slow lane reads from and hands to its node.
pub trait SlowNode {
    /// `getAdvances(n, owner, amount)`.
    fn burn(&self, burn: &Burn) -> anyhow::Result<BranchBurn>;
    /// `getFastLane`.
    fn fast_lane(&self) -> anyhow::Result<FastLaneView>;
    /// `getVaultState`, `getSlotsByOwner(vault)`.
    fn vault(&self, vault: &Address) -> anyhow::Result<VaultView>;
    /// `getEpochAnchor`.
    fn epoch_anchor(&self) -> anyhow::Result<[u8; 32]>;
    /// `getSlotHints(count)`.
    fn slot_hints(&self, count: u32) -> anyhow::Result<Vec<u32>>;
    /// `submitDestinationObject`.
    fn submit(&self, intent: &[u8]) -> anyhow::Result<Value>;
}

impl SlowNode for crate::rpc::NodeClient {
    fn burn(&self, burn: &Burn) -> anyhow::Result<BranchBurn> {
        branch_burn_of(&self.call(
            "getAdvances",
            json!([burn.n, burn.owner.to_bech32(), burn.amount]),
        )?)
    }

    fn fast_lane(&self) -> anyhow::Result<FastLaneView> {
        fast_lane_view_of(&self.call("getFastLane", json!([]))?)
    }

    fn vault(&self, vault: &Address) -> anyhow::Result<VaultView> {
        let state = self.call("getVaultState", json!([]))?;
        let number = |key: &str| {
            state[key]
                .as_u64()
                .ok_or_else(|| anyhow!("getVaultState: {key}"))
        };
        Ok(VaultView {
            tip: number("tip_height")?,
            floor: number("floor")?,
            released_in_window: number("released_in_window")?,
            window_cap: number("window_cap")?,
            slots: crate::fast_release::vault_slots_of(
                &self.call("getSlotsByOwner", json!([vault.to_bech32()]))?,
            )?,
        })
    }

    fn epoch_anchor(&self) -> anyhow::Result<[u8; 32]> {
        crate::base_capture::hex32_of(
            self.call("getEpochAnchor", json!([]))?
                .as_str()
                .context("getEpochAnchor: not a string")?,
        )
    }

    fn slot_hints(&self, count: u32) -> anyhow::Result<Vec<u32>> {
        Ok(serde_json::from_value(self.call("getSlotHints", json!([count]))?)?)
    }

    fn submit(&self, intent: &[u8]) -> anyhow::Result<Value> {
        self.call("submitDestinationObject", json!([hex::encode(intent)]))
    }
}

// ---------------------------------------------------------------------------
// One pass
// ---------------------------------------------------------------------------

/// The newest game Base holds valid that the relay captured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidFound {
    pub game: ValidGame,
    pub proof: GameProof,
    pub capture: Capture,
}

/// The newest of `found` (by factory index, whatever their order) that Base
/// holds valid under the L1 block `l1` (`judge_game` on its proof,
/// `proof_of`), resolved for the defender and captured (`capture_of`).
pub fn newest_valid_game(
    found: &[CapturedGame],
    l1: &FinalizedExecution,
    params: &BaseParams,
    capture_of: &dyn Fn(u64) -> anyhow::Result<Option<Capture>>,
    proof_of: &mut dyn FnMut(&[u8; 20]) -> anyhow::Result<GameProof>,
) -> anyhow::Result<Option<ValidFound>> {
    let mut newest_first: Vec<&CapturedGame> = found.iter().collect();
    newest_first.sort_by(|a, b| b.index.cmp(&a.index));
    for game in newest_first {
        if game.status != GAME_STATUS_DEFENDER_WINS {
            continue;
        }
        let Some(capture) = capture_of(game.l2_block)? else {
            continue;
        };
        let proof = proof_of(&game.game)?;
        if let Ok(valid) = slow_game_under(l1, params, &proof) {
            return Ok(Some(ValidFound {
                game: valid,
                proof,
                capture,
            }));
        }
    }
    Ok(None)
}

/// A burn the slow lane releases this pass, with V2's payee.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wanted {
    pub seen: BurnSeen,
    pub payee: Address,
}

/// What one pass of the slow lane did, by withdrawal index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SlowPassReport {
    /// Found paid on the branch this pass.
    pub paid: Vec<u64>,
    pub fast_pending: Vec<u64>,
    pub in_flight: Vec<u64>,
    pub never_payable: Vec<u64>,
    pub gave_up: Vec<u64>,
    /// No game Base holds valid covers it yet.
    pub waiting: Vec<u64>,
    /// Covered, but beyond one block (budget, count, window): next pass.
    pub held_back: Vec<u64>,
    pub refused: Vec<(u64, String)>,
    /// Built by a dry run, written, not sent.
    pub dry_run: Vec<u64>,
    /// Handed to the node and admitted.
    pub sent: Vec<u64>,
    /// Handed to the node, its answer an error (kept in flight).
    pub not_admitted: Vec<(u64, String)>,
    /// The game the pass released under: its address and Base block.
    pub game: Option<([u8; 20], u64)>,
}

impl SlowPassReport {
    /// One line for the pass's log.
    pub fn summary(&self) -> String {
        let list = |v: &[u64]| {
            v.iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        let firsts = |v: &[(u64, String)]| v.iter().map(|(n, _)| *n).collect::<Vec<_>>();
        format!(
            "slow lane: sent [{}], dry run [{}], in flight [{}], fast pending [{}], waiting for a valid game [{}], \
             held back [{}], refused [{}], not admitted [{}], paid [{}], gave up [{}], never payable [{}]{}",
            list(&self.sent),
            list(&self.dry_run),
            list(&self.in_flight),
            list(&self.fast_pending),
            list(&self.waiting),
            list(&self.held_back),
            list(&firsts(&self.refused)),
            list(&firsts(&self.not_admitted)),
            list(&self.paid),
            list(&self.gave_up),
            list(&self.never_payable),
            match self.game {
                Some((game, l2_block)) => format!(", under game 0x{} (Base block {l2_block})", hex::encode(game)),
                None => String::new(),
            }
        )
    }
}

/// The burns of `burns` the slow lane releases this pass (module docs):
/// each not known paid is read from the node (`getAdvances`, once
/// `getFastLane`), judged ([`judge_burn`]); found paid or given up, the
/// journal says so (stored).
#[allow(clippy::too_many_arguments)]
pub fn slow_wanted(
    node: &dyn SlowNode,
    burns: &BTreeMap<u64, BurnSeen>,
    fast_sent: &BTreeMap<u64, u64>,
    journal: &mut SlowJournal,
    journal_path: &Path,
    cfg: &SlowConfig,
    now: u64,
    report: &mut SlowPassReport,
) -> anyhow::Result<Vec<Wanted>> {
    if cfg.mode == SlowMode::Off {
        return Ok(Vec::new());
    }
    let mut lane: Option<FastLaneView> = None;
    let mut wanted = Vec::new();
    let mut changed = false;
    for seen in burns.values() {
        let burn = seen.burn;
        let entry = journal.entry_of(&burn).cloned();
        match entry.as_ref().map(|e| &e.status) {
            // Paid: never read again.
            Some(SlowStatus::Paid { .. }) => continue,
            // A dry run built it: nothing more to do while it is one.
            Some(SlowStatus::DryRun { .. }) if cfg.mode == SlowMode::DryRun => continue,
            _ => {}
        }
        let branch = node.burn(&burn)?;
        if !branch.active {
            bail!("getAdvances: the return over Base does not govern the node's next block");
        }
        if lane.is_none() {
            lane = Some(node.fast_lane()?);
        }
        let lane = lane.as_ref().expect("read above");
        let attempts = entry.as_ref().map_or(0, |e| e.attempts);
        let settled = |status| SlowEntry {
            owner: burn.owner,
            amount: burn.amount,
            attempts,
            status,
        };
        match judge_burn(&burn, &branch, lane, fast_sent.get(&burn.n).copied(), entry.as_ref(), cfg, now) {
            Verdict::Paid => {
                report.paid.push(burn.n);
                journal
                    .entries
                    .insert(burn.n, settled(SlowStatus::Paid { at: now }));
                changed = true;
            }
            Verdict::GaveUp => {
                if !matches!(entry.as_ref().map(|e| &e.status), Some(SlowStatus::GaveUp { .. })) {
                    report.gave_up.push(burn.n);
                    journal
                        .entries
                        .insert(burn.n, settled(SlowStatus::GaveUp { at: now }));
                    changed = true;
                }
            }
            Verdict::FastPending { .. } => report.fast_pending.push(burn.n),
            Verdict::InFlight { .. } => report.in_flight.push(burn.n),
            Verdict::NeverPayable => report.never_payable.push(burn.n),
            Verdict::DryRunDone => {}
            Verdict::Release => wanted.push(Wanted {
                seen: *seen,
                payee: branch.payee,
            }),
        }
    }
    if changed {
        journal.store(journal_path)?;
    }
    Ok(wanted)
}

/// The slow releases of `wanted` under `valid` (the newest game Base holds
/// valid, read under the finality update `finality`): those whose Base
/// block it covers, by increasing `n`, as many as one block carries; each
/// group built from vault slots no other intent of this relay holds, proven
/// (`prove`), checked ([`verify_slow_intent`]); then, by `cfg.mode`, written
/// to `dry_dir` or handed to the node after the journal says so on disk.
/// FR-3 (audit n°2): nor the slots `fast_held` (the fast lane's releases in flight).
#[allow(clippy::too_many_arguments)]
pub fn slow_release_pass(
    node: &dyn SlowNode,
    rules: &DestinationRules,
    wanted: &[Wanted],
    finality: &[u8],
    valid: Option<&ValidFound>,
    journal: &mut SlowJournal,
    journal_path: &Path,
    dry_dir: &Path,
    cfg: &SlowConfig,
    now: u64,
    prove: &dyn Fn(Vec<TxPage>) -> anyhow::Result<PagedSpendIntent>,
    report: &mut SlowPassReport,
    fast_held: &BTreeSet<u32>,
) -> anyhow::Result<()> {
    if cfg.mode == SlowMode::Off || wanted.is_empty() {
        return Ok(());
    }
    let Some(valid) = valid else {
        report.waiting.extend(wanted.iter().map(|w| w.seen.burn.n));
        return Ok(());
    };
    // One game frees every burn at or before its Base block; the later ones
    // wait for a later game.
    let l2_block = valid.game.l2_block_number;
    let mut covered: Vec<&Wanted> = Vec::new();
    for w in wanted {
        if w.seen.block <= l2_block {
            covered.push(w);
        } else {
            report.waiting.push(w.seen.burn.n);
        }
    }
    if covered.is_empty() {
        return Ok(());
    }
    let burns: Vec<Burn> = covered.iter().map(|w| w.seen.burn).collect();
    let payees: BTreeMap<u64, Address> = covered.iter().map(|w| (w.seen.burn.n, w.payee)).collect();
    let (objects, refused) = assemble_slow(finality, &valid.proof, &valid.capture, &burns, rules)
        .map_err(|why| {
            anyhow!(
                "no slow release under game 0x{} (Base block {l2_block}): {why}",
                hex::encode(valid.game.game)
            )
        })?;
    report.game = Some((objects.game.game, objects.game.l2_block_number));
    report
        .refused
        .extend(refused.into_iter().map(|(n, why)| (n, format!("{why}"))));
    if objects.releases.is_empty() {
        return Ok(());
    }

    // What one block carries: the shared objects once, the releases by
    // increasing `n` (the template's order, never jumping one that waits),
    // within the section, `max_releases_per_block` and the V6 window's room.
    let vault = node.vault(&rules.vault.vault)?;
    let gate = rules.vault.at(vault.tip.saturating_add(1));
    let shared = shared_objects(&objects);
    let mut room = vault
        .window_cap
        .saturating_sub(vault.released_in_window)
        .saturating_sub(in_flight_amount(journal, cfg, now));
    let mut batch: Vec<VaultRelease> = Vec::new();
    let mut stopped = false;
    for release in &objects.releases {
        if fit_in_one_block(&shared, std::slice::from_ref(release), 1) == 0 {
            report
                .refused
                .push((release.n, "alone beyond one block's destination section".into()));
            continue;
        }
        let mut candidate = batch.clone();
        candidate.push(release.clone());
        if stopped
            || fit_in_one_block(&shared, &candidate, gate.max_releases_per_block) < candidate.len()
            || release.amount > room
        {
            stopped = true;
            report.held_back.push(release.n);
            continue;
        }
        room -= release.amount;
        batch = candidate;
    }
    if batch.is_empty() {
        return Ok(());
    }

    let anchor = node.epoch_anchor()?;
    let hints = node.slot_hints(u32::try_from(3 * batch.len())?)?;
    if hints.len() < 3 * batch.len() {
        bail!("getSlotHints: {} hints for {} groups", hints.len(), batch.len());
    }
    // FR-3 (audit n°2): never the slots of the fast lane's releases in flight either
    let mut taken: BTreeSet<u32> = fast_held.clone();
    for (i, release) in batch.iter().enumerate() {
        let n = release.n;
        let payee = payees[&n];
        let held = held_slots(journal, n);
        let free: Vec<VaultSlot> = vault
            .slots
            .iter()
            .copied()
            .filter(|s| !held.contains(&s.slot_index) && !taken.contains(&s.slot_index))
            .collect();
        let Some(inputs) = select_vault_slots(&free, vault.floor, release.amount) else {
            report.refused.push((
                n,
                format!("the vault's free slots do not cover {} µJTM", release.amount),
            ));
            continue;
        };
        let built = slow_release_group(release, &payee, &inputs, anchor, &hints[3 * i..3 * i + 3], &gate, vault.floor)
            .and_then(|(pages, _)| prove(pages))
            .and_then(|payment| {
                let txid = hex::encode(payment.logical_txid().0);
                Ok((slow_release_intent(&objects, release, payment)?, txid))
            })
            .and_then(|(bytes, txid)| {
                verify_slow_intent(&bytes, rules, &gate, vault.floor, &payee)?;
                Ok((bytes, txid))
            });
        let (bytes, txid) = match built {
            Ok(built) => built,
            Err(e) => {
                report.refused.push((n, format!("{e:#}")));
                continue;
            }
        };
        taken.extend(inputs.iter().map(|s| s.slot_index));
        let attempts = journal
            .entries
            .get(&n)
            .filter(|e| e.owner == release.owner && e.amount == release.amount)
            .map_or(0, |e| e.attempts);
        let entry = |attempts, status| SlowEntry {
            owner: release.owner,
            amount: release.amount,
            attempts,
            status,
        };
        match cfg.mode {
            SlowMode::Off => unreachable!("returned above"),
            SlowMode::DryRun => {
                std::fs::create_dir_all(dry_dir)?;
                crate::write_atomic(
                    &dry_dir.join(format!("slow-release-{n}.hex")),
                    hex::encode(&bytes).as_bytes(),
                )?;
                let mut next = journal.clone();
                next.entries.insert(
                    n,
                    entry(attempts, SlowStatus::DryRun { at: now, l2_block, txid }),
                );
                next.store(journal_path)?;
                *journal = next;
                report.dry_run.push(n);
            }
            SlowMode::Send => {
                // Write-ahead: on disk before the node is called; a journal
                // that cannot be written sends nothing.
                let mut next = journal.clone();
                next.entries.insert(
                    n,
                    entry(
                        attempts.saturating_add(1),
                        SlowStatus::Submitted {
                            at: now,
                            l2_block,
                            txid,
                            admitted: false,
                            slots: inputs.iter().map(|s| s.slot_index).collect(),
                        },
                    ),
                );
                next.store(journal_path)?;
                *journal = next;
                match node.submit(&bytes) {
                    Ok(_) => {
                        if let Some(SlowEntry {
                            status: SlowStatus::Submitted { admitted, .. },
                            ..
                        }) = journal.entries.get_mut(&n)
                        {
                            *admitted = true;
                        }
                        journal.store(journal_path)?;
                        report.sent.push(n);
                    }
                    // Not admitted, or its answer lost: in flight until
                    // `resubmit_secs` (never handed in twice meanwhile).
                    Err(e) => report.not_admitted.push((n, format!("{e:#}"))),
                }
            }
        }
    }
    Ok(())
}
