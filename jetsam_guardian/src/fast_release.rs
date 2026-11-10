// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The fast lane of the return over Base, relay side, for real (D127, D130,
//! RETOUR-RAPIDE SPEC §10): the whole intent of one fast release, handed to a
//! node.
//!
//! Under one finalized L1 block (the block a `LightClientFinalityUpdate`
//! names), the relay:
//!
//! 1. judges the game as the dry run does ([`crate::fast::judge`]) and signs
//!    its statement with its fast-lane key (object 7);
//! 2. reads the game's proof by `eth_getProof` at that block, the slots the
//!    rule reads ([`game_proof_slots`], object 5), and checks it natively
//!    ([`read_game_facts`], [`judge_game_fast`]) under the update's root and
//!    time, against the statement;
//! 3. opens the game's root claim with the Base block its capture holds
//!    (object 6) and reads, under the Base state root, the bridge's account,
//!    its `genesisId` (object 3) and `withdrawals[n]` (object 4): each proof
//!    checked natively ([`verify_bridge_account`], [`verify_withdrawal`]);
//! 4. builds the release group from the vault's slots: the payee (the burn's
//!    owner, or the creditor of its advance) receives `amount − R`, then the
//!    fast lane's marker, then the change to the vault; checked against the
//!    block's gate ([`check_release_group_marked`]) before it is proven with
//!    the vault's published secret;
//! 5. hands the intent (`JTMDIN01`, kind 4) to the node's
//!    `submitDestinationObject`, or, in a dry run, writes it and sends
//!    nothing.
//!
//! Every refusal is an `ALERT FAST_RELEASE` line and nothing is sent. The
//! light-client update itself (and the committee updates that let a node
//! verify it) is an input: no tool of this repository fetches or posts it
//! yet outside the rehearsal.

use anyhow::{anyhow, bail, Context};
use serde_json::{json, Value};

use jetsam_chain::consensus::destination::base_read::{
    encode_game_proof, encode_output_root_preimage, game_proof_slots, judge_game_fast,
    open_output_root, read_game_facts, BaseGameError, BaseParams, BaseReturnError, ContractProof,
    GameProof, OutputRootPreimage, ValidGame,
};
use jetsam_chain::consensus::destination::fast::{
    fast_release_marker, verify_attestation, FastAttestation, FastSignatureError, FastStatement,
};
use jetsam_chain::consensus::destination::l1_read::{
    genesis_slot, verify_bridge_account, verify_withdrawal, withdrawal_slot, L1BridgeParams,
    L1ReadError,
};
use jetsam_chain::consensus::destination::light_client::{
    finalized_execution, FinalizedExecution, LightClientError,
};
use jetsam_chain::consensus::destination::ssz::LightClientFinalityUpdate;
use jetsam_chain::consensus::destination::state::DestinationRules;
use jetsam_chain::consensus::params::is_coinbase_creation_id;
use jetsam_chain::consensus::vault::{
    check_release_group_marked, encode_bridge_account_proof, VaultError, VaultRelease, VaultRules,
    VAULT_SECRET,
};
use jetsam_mempool::destination::{BaseReleaseProofs, DestinationIntent, ReleaseIntent};
use jetsam_poseidon2b::primitives::{Address, SpendSecret};
use jetsam_tx::{
    output_bitmap_bit, PagedSpendIntent, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

use crate::base_capture::{hex20, BaseRpc, Capture};
use crate::fast::StatementDomain;

#[cfg(test)]
#[path = "fast_release_tests.rs"]
mod tests;

/// One burn on Base: `withdrawals[n] = keccak(owner ‖ amount)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Burn {
    pub n: u64,
    pub owner: Address,
    pub amount: u64,
}

/// One slot of the vault as `getSlotsByOwner` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaultSlot {
    pub slot_index: u32,
    pub amount: u64,
    pub creation_id: u64,
}

/// Why the relay assembles no fast release (each an `ALERT FAST_RELEASE`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AssemblyRefusal {
    /// The finality update's payload does not decode.
    FinalityPayload,
    /// Its execution header does not name a block (`finalized_execution`).
    FinalityExecution(LightClientError),
    /// Its execution header carries no time.
    NoL1Time,
    /// The game does not read, or is not one the fast lane takes, under the
    /// update's root and time.
    Game(BaseGameError),
    /// The attestation names another game, L2 block, root, network, chain or
    /// bridge than the ones read.
    NotTheStatement,
    /// The attestation does not verify under the build's release key.
    Attestation(FastSignatureError),
    /// This build has no bridge (`L1_BRIDGE` unset).
    BridgeUnset,
    /// The capture is of another Base block or account than the game's.
    CaptureOfAnotherBlock {
        capture: u64,
        l2_block: u64,
    },
    CaptureOfAnotherAccount,
    /// The capture lacks the slot (`genesisId` or `withdrawals[n]`).
    SlotNotCaptured(&'static str),
    /// The game's claim does not open with the capture's block.
    OutputRoot(BaseReturnError),
    /// The bridge read under the Base state root (code, genesis, the burn).
    Bridge(L1ReadError),
}

impl std::fmt::Display for AssemblyRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// The objects of one fast release, each checked natively.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FastReleaseObjects {
    /// The L1 block they are read under.
    pub l1: FinalizedExecution,
    pub game: ValidGame,
    /// Object payloads: the finality update, the game proof, the output root
    /// preimage, the bridge account proof (with its `genesisId`), the
    /// attestation.
    pub finality: Vec<u8>,
    pub game_proof: Vec<u8>,
    pub output_preimage: Vec<u8>,
    pub account_proof: Vec<u8>,
    pub attestation: Vec<u8>,
    /// Object 4: the release, with the proof of `withdrawals[n]`.
    pub release: VaultRelease,
}

/// The game under the L1 block `l1` (its root and time): read from `proof`,
/// judged by the fast lane, and exactly what `attestation` states for
/// `domain`, signed by `release_key`.
pub fn game_under(
    l1: &FinalizedExecution,
    params: &BaseParams,
    proof: &GameProof,
    attestation: &FastAttestation,
    domain: &StatementDomain,
    release_key: Option<&[u8; 48]>,
) -> Result<ValidGame, AssemblyRefusal> {
    let l1_time = l1.timestamp.ok_or(AssemblyRefusal::NoL1Time)?;
    let facts = read_game_facts(l1.state_root, params, proof).map_err(AssemblyRefusal::Game)?;
    let game = judge_game_fast(&facts, l1_time, params).map_err(AssemblyRefusal::Game)?;
    let expected = FastStatement {
        network: domain.network,
        l1_chain_id: domain.l1_chain_id,
        base_chain_id: domain.base_chain_id,
        bridge: domain.bridge,
        game: game.game,
        l2_block: game.l2_block_number,
        root_claim: game.root_claim,
    };
    if attestation.statement != expected {
        return Err(AssemblyRefusal::NotTheStatement);
    }
    verify_attestation(attestation, release_key).map_err(AssemblyRefusal::Attestation)?;
    Ok(game)
}

/// The Base side of a release under `game`: its claim opened with the
/// capture's block (the preimage), the bridge's account and `genesisId`
/// (the account proof payload) and the release of `burn` (with the proof of
/// `withdrawals[n]`), each checked under the Base state root.
pub fn bridge_under(
    game: &ValidGame,
    capture: &Capture,
    burn: &Burn,
    bridge: &L1BridgeParams,
) -> Result<(OutputRootPreimage, Vec<u8>, VaultRelease), AssemblyRefusal> {
    if capture.block != game.l2_block_number {
        return Err(AssemblyRefusal::CaptureOfAnotherBlock {
            capture: capture.block,
            l2_block: game.l2_block_number,
        });
    }
    if capture.address != bridge.address {
        return Err(AssemblyRefusal::CaptureOfAnotherAccount);
    }
    let proof_of = |key: [u8; 32], name: &'static str| {
        capture
            .storage
            .iter()
            .find(|s| s.key == key)
            .map(|s| s.proof.clone())
            .ok_or(AssemblyRefusal::SlotNotCaptured(name))
    };
    let genesis = proof_of(genesis_slot(bridge.genesis_id_slot), "genesisId")?;
    let withdrawal = proof_of(
        withdrawal_slot(burn.n, bridge.withdrawals_slot),
        "withdrawals[n]",
    )?;
    let preimage = OutputRootPreimage {
        version: [0u8; 32],
        state_root: capture.state_root,
        message_passer_storage_root: capture.withdrawals_root,
        block_hash: capture.block_hash,
    };
    let state_root =
        open_output_root(&game.root_claim, &preimage).map_err(AssemblyRefusal::OutputRoot)?;
    verify_bridge_account(state_root, bridge, &capture.account_proof, &genesis)
        .map_err(AssemblyRefusal::Bridge)?;
    verify_withdrawal(
        state_root,
        bridge,
        &capture.account_proof,
        burn.n,
        &burn.owner.0,
        burn.amount,
        &withdrawal,
    )
    .map_err(AssemblyRefusal::Bridge)?;
    Ok((
        preimage,
        encode_bridge_account_proof(&capture.account_proof, &genesis),
        VaultRelease {
            n: burn.n,
            owner: burn.owner,
            amount: burn.amount,
            storage_proof: withdrawal,
        },
    ))
}

/// All the objects of a fast release of `burn`: the finality update's
/// payload `finality`, the game's `proof`, the capture of its Base block, the
/// relay's `attestation`, under the build's `rules`.
pub fn assemble(
    finality: &[u8],
    proof: &GameProof,
    capture: &Capture,
    burn: &Burn,
    attestation: &FastAttestation,
    rules: &DestinationRules,
) -> Result<FastReleaseObjects, AssemblyRefusal> {
    let update = LightClientFinalityUpdate::decode_payload(finality)
        .map_err(|_| AssemblyRefusal::FinalityPayload)?;
    let l1 = finalized_execution(&update).map_err(AssemblyRefusal::FinalityExecution)?;
    let bridge = rules.bridge.ok_or(AssemblyRefusal::BridgeUnset)?;
    let fast = &rules.base_return.fast;
    let domain = StatementDomain {
        network: jetsam_chain::consensus::guardian::network_id(),
        l1_chain_id: fast.l1_chain_id,
        base_chain_id: fast.base_chain_id,
        bridge: bridge.address,
    };
    let game = game_under(
        &l1,
        &rules.base_return.base,
        proof,
        attestation,
        &domain,
        fast.release_key.as_ref(),
    )?;
    let (preimage, account_proof, release) = bridge_under(&game, capture, burn, &bridge)?;
    Ok(FastReleaseObjects {
        l1,
        game,
        finality: finality.to_vec(),
        game_proof: encode_game_proof(proof),
        output_preimage: encode_output_root_preimage(&preimage),
        account_proof,
        attestation: attestation.encode_payload(),
        release,
    })
}

/// The vault slots a release of `amount` spends: created after `floor` and
/// not by a coinbase (V1-b), largest first; `None` if they do not cover it.
pub fn select_vault_slots(slots: &[VaultSlot], floor: u64, amount: u64) -> Option<Vec<VaultSlot>> {
    let mut candidates: Vec<VaultSlot> = slots
        .iter()
        .copied()
        .filter(|s| s.creation_id > floor && !is_coinbase_creation_id(s.creation_id))
        .collect();
    candidates.sort_by(|a, b| {
        b.amount
            .cmp(&a.amount)
            .then(a.slot_index.cmp(&b.slot_index))
    });
    let mut picked = Vec::new();
    let mut total = 0u64;
    for slot in candidates {
        if total >= amount {
            break;
        }
        total = total.checked_add(slot.amount)?;
        picked.push(slot);
    }
    (total >= amount).then_some(picked)
}

/// The pages of one logical transaction from `input_owner`: `inputs` eight
/// per page, `outputs` two per page (each on the slot hint of its rank), the
/// fee on the first page.
fn pack_group(
    input_owner: Address,
    inputs: &[VaultSlot],
    outputs: &[(Address, u64)],
    fee: u64,
    epoch_anchor: [u8; 32],
    slot_hints: &[u32],
) -> anyhow::Result<Vec<TxPage>> {
    if slot_hints.len() < outputs.len() {
        bail!("{} outputs, {} slot hints", outputs.len(), slot_hints.len());
    }
    let page_count = inputs
        .len()
        .div_ceil(TX_INPUTS)
        .max(outputs.len().div_ceil(TX_OUTPUTS))
        .max(1);
    let mut pages = Vec::with_capacity(page_count);
    for page in 0..page_count {
        let mut page_inputs = [TxInput::dummy(); TX_INPUTS];
        let mut page_outputs = [TxOutput::dummy(); TX_OUTPUTS];
        let mut bitmap = 0u16;
        for (slot, input) in page_inputs.iter_mut().enumerate() {
            if let Some(u) = inputs.get(page * TX_INPUTS + slot) {
                *input = TxInput {
                    slot_index: u.slot_index,
                    amount: u.amount,
                    creation_id: u.creation_id,
                };
                bitmap |= 1u16 << slot;
            }
        }
        for (slot, output) in page_outputs.iter_mut().enumerate() {
            let rank = page * TX_OUTPUTS + slot;
            if let Some((owner, amount)) = outputs.get(rank) {
                *output = TxOutput {
                    slot_index: slot_hints[rank],
                    amount: *amount,
                    owner: *owner,
                };
                bitmap |= output_bitmap_bit(slot);
            }
        }
        if page == 0 {
            bitmap |= PAGED_SPEND_START_BIT;
        }
        if page + 1 == page_count {
            bitmap |= PAGED_SPEND_END_BIT;
        }
        pages.push(
            TxPage::new(TxBody {
                epoch_anchor,
                fee: if page == 0 { fee } else { 0 },
                input_owner,
                inputs: page_inputs,
                outputs: page_outputs,
                validity_bitmap: bitmap,
                is_coinbase: false,
            })
            .map_err(|e| anyhow!("page: {e:?}"))?,
        );
    }
    Ok(pages)
}

/// The release group of a fast release of `release` under the game of L2
/// block `l2_block`: `payee` receives `amount − R`, then the fast lane's
/// marker, then the change to the vault; its fee is `R` (D59). Checked
/// against the block's gate before it is returned, with `R`.
#[allow(clippy::too_many_arguments)]
pub fn fast_release_group(
    release: &VaultRelease,
    l2_block: u64,
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
    let marker = fast_release_marker(release.n, &release.owner, release.amount, l2_block);
    let mut outputs = vec![(*payee, paid), (marker, 0)];
    if change > 0 {
        outputs.push((rules.vault, change));
    }
    let pages = pack_group(rules.vault, inputs, &outputs, r, epoch_anchor, slot_hints)?;
    check_release_group_marked(&pages, release, rules, floor, payee, &marker)
        .map_err(|e: VaultError| anyhow!("the block's gate refuses the group: {e:?}"))?;
    Ok((pages, r))
}

/// The vault group proven with the vault's published secret.
pub fn prove_vault_group(pages: Vec<TxPage>) -> anyhow::Result<PagedSpendIntent> {
    let bundle = jetsam_gkr::prove_paged_spend_authorization(
        &pages,
        jetsam_gkr::OwnerAuthWitness::new(SpendSecret::from_bytes(VAULT_SECRET)),
    )
    .map_err(|e| anyhow!("the vault group's proof: {e:?}"))?;
    PagedSpendIntent::new(
        pages,
        bundle
            .to_bytes()
            .map_err(|e| anyhow!("the vault group's proof: {e:?}"))?,
    )
    .map_err(|e| anyhow!("the vault group: {e:?}"))
}

/// The bytes `submitDestinationObject` takes: the fast release of `objects`
/// paid by `payment`.
pub fn fast_release_intent(
    objects: &FastReleaseObjects,
    payment: PagedSpendIntent,
) -> anyhow::Result<Vec<u8>> {
    DestinationIntent::Release(ReleaseIntent {
        release: objects.release.clone(),
        finality: objects.finality.clone(),
        account_proof: objects.account_proof.clone(),
        payment,
        base: Some(BaseReleaseProofs {
            game_proof: objects.game_proof.clone(),
            output_preimage: objects.output_preimage.clone(),
        }),
        fast: Some(objects.attestation.clone()),
    })
    .encode()
    .map_err(|e| anyhow!("the intent: {e}"))
}

// ---------------------------------------------------------------------------
// Reading the game's proof (eth_call, eth_getProof)
// ---------------------------------------------------------------------------

/// One contract's proof out of an `eth_getProof` reply for `address` and
/// `keys`, in their order (the RPC must answer exactly them).
pub fn contract_proof_of(
    reply: &Value,
    address: &[u8; 20],
    keys: &[[u8; 32]],
) -> anyhow::Result<ContractProof> {
    if hex20(reply["address"].as_str().unwrap_or_default())? != *address {
        bail!("the RPC served the proof of another account");
    }
    let served = reply["storageProof"]
        .as_array()
        .ok_or_else(|| anyhow!("storageProof: not a list"))?;
    if served.len() != keys.len() {
        bail!(
            "the RPC served {} slot proofs for {}",
            served.len(),
            keys.len()
        );
    }
    let mut slots = Vec::with_capacity(keys.len());
    for (key, entry) in keys.iter().zip(served) {
        if crate::base_capture::hex32_of(entry["key"].as_str().unwrap_or_default())? != *key {
            bail!("the RPC served the proof of another slot");
        }
        slots.push(crate::base_capture::nodes(
            &entry["proof"],
            "storage proof",
        )?);
    }
    Ok(ContractProof {
        account: crate::base_capture::nodes(&reply["accountProof"], "accountProof")?,
        slots,
    })
}

fn dynamic_bytes(answer: &[u8]) -> anyhow::Result<Vec<u8>> {
    let word = |i: usize| -> anyhow::Result<usize> {
        let w = answer
            .get(32 * i..32 * (i + 1))
            .ok_or_else(|| anyhow!("answer too short"))?;
        if w[..24] != [0u8; 24] {
            bail!("a length beyond 64 bits");
        }
        Ok(u64::from_be_bytes(w[24..].try_into().expect("8")) as usize)
    };
    let offset = word(0)?;
    if offset % 32 != 0 {
        bail!("an odd offset");
    }
    let len = word(offset / 32)?;
    answer
        .get(offset + 32..offset + 32 + len)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| anyhow!("answer too short"))
}

/// The proof of the game `game` under the L1 block `block`: its immutable
/// arguments rebuilt from its getters (`eth_call`), then the slots the rule
/// reads of the five contracts (`eth_getProof`).
pub fn read_game_proof(
    l1: &dyn BaseRpc,
    params: &BaseParams,
    game: &[u8; 20],
    block: u64,
) -> anyhow::Result<GameProof> {
    let tag = format!("0x{block:x}");
    let get = |to: &[u8; 20], signature: &str| crate::fast::call(l1, to, signature, &[], &tag);
    let word = |to: &[u8; 20], signature: &str| -> anyhow::Result<[u8; 32]> {
        get(to, signature)?
            .get(..32)
            .map(|w| w.try_into().expect("32"))
            .ok_or_else(|| anyhow!("{signature}: answer too short"))
    };
    // creator (20) ‖ rootClaim (32) ‖ l1Head (32) ‖ extraData (L2 block 32 ‖ parent 20 ‖ roots).
    let mut args = word(game, "gameCreator()")?[12..].to_vec();
    args.extend_from_slice(&word(game, "rootClaim()")?);
    args.extend_from_slice(&word(game, "l1Head()")?);
    args.extend(dynamic_bytes(&get(game, "extraData()")?)?);
    let portal: [u8; 20] = word(&params.system_config, "optimismPortal()")?[12..]
        .try_into()
        .expect("20");
    let slots = game_proof_slots(params, game, &args, &portal).ok_or_else(|| {
        anyhow!("the game's arguments are not of a pinned implementation's length")
    })?;
    let prove = |address: &[u8; 20], keys: &[[u8; 32]]| -> anyhow::Result<ContractProof> {
        let reply = l1.rpc(
            "eth_getProof",
            json!([
                format!("0x{}", hex::encode(address)),
                keys.iter()
                    .map(|k| format!("0x{}", hex::encode(k)))
                    .collect::<Vec<_>>(),
                tag
            ]),
        )?;
        contract_proof_of(&reply, address, keys)
    };
    Ok(GameProof {
        game: *game,
        anchor_state_registry: prove(&params.anchor_state_registry, &slots.anchor_state_registry)?,
        dispute_game_factory: prove(&params.dispute_game_factory, &slots.dispute_game_factory)?,
        game_account: prove(game, &slots.game)?,
        system_config: prove(&params.system_config, &slots.system_config)?,
        superchain_config: prove(&params.superchain_config, &slots.superchain_config)?,
        args,
    })
}

// ---------------------------------------------------------------------------
// The node (its public reads and submitDestinationObject)
// ---------------------------------------------------------------------------

/// The vault's slots out of a `getSlotsByOwner` reply.
pub fn vault_slots_of(reply: &Value) -> anyhow::Result<Vec<VaultSlot>> {
    Ok(reply
        .as_array()
        .context("getSlotsByOwner: not a list")?
        .iter()
        .filter(|s| s["empty"].as_bool() != Some(true))
        .filter_map(|s| {
            Some(VaultSlot {
                slot_index: u32::try_from(s["slot_index"].as_u64()?).ok()?,
                amount: s["value"].as_u64()?,
                creation_id: s["creation_id"].as_u64()?,
            })
        })
        .collect())
}

/// What the node gives the release group: its tip, the vault's floor and
/// slots, the payee of the burn, an epoch anchor and three slot hints.
struct NodeReads {
    tip: u64,
    floor: u64,
    slots: Vec<VaultSlot>,
    payee: Address,
    anchor: [u8; 32],
    hints: Vec<u32>,
}

fn read_node(
    node: &crate::rpc::NodeClient,
    burn: &Burn,
    vault: &Address,
) -> anyhow::Result<NodeReads> {
    let tip = node
        .call("blockCount", json!([]))?
        .as_u64()
        .context("blockCount: not a number")?;
    let floor = node.call("getVaultState", json!([]))?["floor"]
        .as_u64()
        .context("getVaultState: floor")?;
    let slots = vault_slots_of(&node.call("getSlotsByOwner", json!([vault.to_bech32()]))?)?;
    let advances = node.call(
        "getAdvances",
        json!([burn.n, burn.owner.to_bech32(), burn.amount]),
    )?;
    if advances["active"].as_bool() != Some(true) {
        bail!("getAdvances: the return over Base does not govern the node's next block");
    }
    let payee = Address::parse(
        advances["payee"]
            .as_str()
            .context("getAdvances: no payee")?,
    )
    .map_err(|e| anyhow!("getAdvances: payee {e:?}"))?;
    let anchor = crate::base_capture::hex32_of(
        node.call("getEpochAnchor", json!([]))?
            .as_str()
            .context("getEpochAnchor: not a string")?,
    )?;
    let hints: Vec<u32> = serde_json::from_value(node.call("getSlotHints", json!([3]))?)?;
    Ok(NodeReads {
        tip,
        floor,
        slots,
        payee,
        anchor,
        hints,
    })
}

// ---------------------------------------------------------------------------
// jg-relay fast-release
// ---------------------------------------------------------------------------

/// `jg-relay fast-release …`: one fast release, assembled, checked and
/// handed to the node (or written, `--dry-run`).
pub fn fast_release_cli(args: &[String]) -> anyhow::Result<()> {
    let usage = || {
        anyhow!(
            "fast-release --key <secret file> --l1 <url> --base <url> --base <url> [--base <url>…] --node <url> \
             --finality <file: hex of the finality update's payload> --game-index <i> --n <n> --owner <address> \
             --amount <µJTM> [--capture-dir <dir>] [--dry-run --out <file>]"
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
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let out = value("--out").map(std::path::PathBuf::from);
    if dry_run && out.is_none() {
        return Err(usage());
    }
    let say = |line: &str| println!("{} jg-relay fast-release {line}", crate::now_unix());
    let alert = |line: String| -> anyhow::Error {
        say(&format!("ALERT FAST_RELEASE {line}"));
        anyhow!("fast-release: {line}")
    };
    let rules = DestinationRules::CONSENSUS;
    let params = rules.base_return.base;
    let bridge = rules.bridge.ok_or_else(|| {
        alert("this build has no bridge (L1_BRIDGE unset): nothing to release".into())
    })?;
    // The relay's key: the build's FAST_RELEASE_KEY (a dry run may use another: nothing is sent).
    let secret =
        crate::fast::load_fast_secret(std::path::Path::new(&value("--key").ok_or_else(usage)?))?;
    let public = secret.sk_to_pk().compress();
    match rules.base_return.fast.release_key {
        Some(key) if key == public => {}
        _ if dry_run => say(&format!(
            "dry run: the key 0x{} is not this build's FAST_RELEASE_KEY (nothing is sent)",
            hex::encode(public)
        )),
        None => return Err(alert("this build has no FAST_RELEASE_KEY".into())),
        Some(_) => {
            return Err(alert(
                "the key given is not this build's FAST_RELEASE_KEY".into(),
            ))
        }
    }
    let l1 = crate::l1::L1::new(&value("--l1").ok_or_else(usage)?);
    let bases: Vec<crate::l1::L1> = values("--base")
        .iter()
        .map(|u| crate::l1::L1::new(u))
        .collect();
    let node = crate::rpc::NodeClient::fast_release(
        &value("--node").ok_or_else(usage)?,
        crate::rpc::Egress::Clearnet,
    );
    let index: u64 = value("--game-index").ok_or_else(usage)?.parse()?;
    let burn = Burn {
        n: value("--n").ok_or_else(usage)?.parse()?,
        owner: Address::parse(&value("--owner").ok_or_else(usage)?)
            .map_err(|e| anyhow!("--owner: {e:?}"))?,
        amount: value("--amount").ok_or_else(usage)?.parse()?,
    };
    let domain = StatementDomain {
        network: jetsam_chain::consensus::guardian::network_id(),
        l1_chain_id: rules.base_return.fast.l1_chain_id,
        base_chain_id: rules.base_return.fast.base_chain_id,
        bridge: bridge.address,
    };
    // 1. The L1 block: the one the finality update names (the node verifies its signature).
    let finality_path = value("--finality").ok_or_else(usage)?;
    let finality = crate::base_capture::hex_bytes(std::fs::read_to_string(&finality_path)?.trim())?;
    let update = LightClientFinalityUpdate::decode_payload(&finality)
        .map_err(|e| alert(format!("the finality update does not decode: {e:?}")))?;
    let l1_block = finalized_execution(&update).map_err(|e| {
        alert(format!(
            "the finality update names no execution block: {e:?}"
        ))
    })?;
    let served = l1.rpc(
        "eth_getBlockByNumber",
        json!([format!("0x{:x}", l1_block.block_number), false]),
    )?;
    if crate::base_capture::hex32_of(served["stateRoot"].as_str().unwrap_or_default())?
        != l1_block.state_root
    {
        return Err(alert(format!(
            "the L1 RPC serves another block {} than the finality update's",
            l1_block.block_number
        )));
    }
    // 2. The game, judged as the dry run judges it, then signed.
    let game = crate::fast::read_game(&l1, &params, index, l1_block.block_number)?;
    let mut views = Vec::with_capacity(bases.len());
    for base in &bases {
        match crate::fast::read_base(base, game.l2_block) {
            Ok(view) => views.push(view),
            Err(e) => say(&format!("ALERT FAST_RELEASE BASE_RPC {}: {e:#}", base.url)),
        }
    }
    let mut cache = std::collections::BTreeMap::new();
    let ancestors = crate::fast::read_ancestors(
        &l1,
        &params,
        game.parent,
        "latest",
        crate::now_unix(),
        &mut cache,
    )?;
    let statement: FastStatement = crate::fast::judge(&game, &params, &views, &ancestors, &domain)
        .map_err(|why| alert(format!("game #{index} not signed: {why}")))?;
    let attestation = crate::fast::sign(&statement, &secret);
    // 3. Its proof under the update's block; the capture of its Base block.
    let proof = read_game_proof(&l1, &params, &game.game, l1_block.block_number)?;
    let wanted = crate::base_capture::capture_slots(
        bridge.genesis_id_slot,
        bridge.withdrawals_slot,
        &[burn.n],
    );
    let kept = match value("--capture-dir") {
        Some(dir) => crate::base_capture::load(std::path::Path::new(&dir), game.l2_block)?,
        None => None,
    };
    let capture = match kept {
        Some(c) if wanted.iter().all(|k| c.storage.iter().any(|s| s.key == *k)) => c,
        _ => {
            let base = bases.first().ok_or_else(usage)?;
            crate::base_capture::capture_at(base, &bridge.address, &wanted, game.l2_block).map_err(
                |e| {
                    alert(format!(
                        "no capture of Base block {} with withdrawals[{}]: {e:#}",
                        game.l2_block, burn.n
                    ))
                },
            )?
        }
    };
    let objects = assemble(&finality, &proof, &capture, &burn, &attestation, &rules)
        .map_err(|why| alert(format!("game #{index}, burn {}: {why}", burn.n)))?;
    // 4. The release group from the vault.
    let reads = read_node(&node, &burn, &rules.vault.vault)?;
    let vault_rules = rules.vault.at(reads.tip.saturating_add(1));
    let inputs = select_vault_slots(&reads.slots, reads.floor, burn.amount).ok_or_else(|| {
        alert(format!(
            "the vault's slots do not cover {} µJTM",
            burn.amount
        ))
    })?;
    let (pages, fee) = fast_release_group(
        &objects.release,
        objects.game.l2_block_number,
        &reads.payee,
        &inputs,
        reads.anchor,
        &reads.hints,
        &vault_rules,
        reads.floor,
    )
    .map_err(|e| alert(format!("the release group: {e:#}")))?;
    let payment = prove_vault_group(pages)?;
    let txid = hex::encode(payment.logical_txid().0);
    let intent = fast_release_intent(&objects, payment)?;
    let summary = json!({
        "game": format!("0x{}", hex::encode(objects.game.game)),
        "l2_block": objects.game.l2_block_number,
        "l1_block": objects.l1.block_number,
        "n": burn.n,
        "amount": burn.amount,
        "payee": reads.payee.to_bech32(),
        "release_fee": fee,
        "vault_inputs": inputs.len(),
        "txid": txid,
        "intent_bytes": intent.len(),
    });
    // 5. To the node, or nowhere.
    if dry_run {
        let out = out.expect("checked above");
        std::fs::write(&out, hex::encode(&intent))?;
        say(&format!(
            "DRY RUN, NOT SENT: {summary} written to {}",
            out.display()
        ));
        return Ok(());
    }
    match node.call("submitDestinationObject", json!([hex::encode(&intent)])) {
        Ok(admitted) => {
            say(&format!("SENT fast release {summary}: {admitted}"));
            Ok(())
        }
        Err(e) => Err(alert(format!(
            "the node refused fast release {summary}: {e:#}"
        ))),
    }
}
