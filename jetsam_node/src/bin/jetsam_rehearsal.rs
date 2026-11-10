// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jetsam-rehearsal` — **PRIVATE REHEARSAL TOOL** (`rehearsal-v1-7` builds
//! only, never distributed, T7.2).
//!
//! What the rehearsal's synthetic L1 (`jetsam_chain::consensus::destination::
//! rehearsal_l1`) and its test guardian key (`…::guardian::rehearsal`) would
//! provide, made into intents a rehearsal node takes over its RPC:
//!
//! * `committee-update`: the anchor update or a rotation, with the
//!   transaction paying its `JTMLCU01` marker (`submitDestinationObject`);
//! * `release`: the release of a burn `(n, owner, amount)` of a burns file
//!   (any `n`, any order), its finality update, the bridge account proof,
//!   the storage proof, and the release group that spends the vault at its
//!   **published** secret (`submitDestinationObject`);
//! * `deposit`: a deposit to the vault with its `JTMBRDG1` marker
//!   (`submitTxIntent`);
//! * `attest`: the test-key attestation of a block of the node's branch, with
//!   the transaction paying its marker (`submitGuardianObject`);
//! * `actor`, `l1-state`, `attest-body`: what they compute, printed.
//!
//! An **actor** is a named rehearsal key (`actor_secret`): the miner pays it
//! (`--miner-address`), it pays the markers and fees. Every key here is a
//! TEST key derived from a public label: nothing it holds has value.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context};
use clap::{Parser, Subcommand};
use serde_json::{json, Value};

use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::destination::rehearsal_l1::{
    anchor_update, bridge_state, committee_update, finality_update, Burn,
};
use jetsam_chain::consensus::destination::state::committee_update_marker;
use jetsam_chain::consensus::difficulty::{add_work, block_work};
use jetsam_chain::consensus::guardian::rehearsal::rehearsal_attestation;
use jetsam_chain::consensus::guardian::GuardianAttestation;
use jetsam_chain::consensus::params::is_coinbase_creation_id;
use jetsam_chain::consensus::pow::block_id;
use jetsam_chain::consensus::vault::{VaultRelease, VaultRules, VAULT_ADDRESS, VAULT_SECRET};
use jetsam_gkr::{prove_paged_spend_authorization, OwnerAuthWitness};
use jetsam_mempool::destination::{CommitteeUpdateIntent, DestinationIntent, ReleaseIntent};
use jetsam_mempool::guardian::GuardianIntent;
use jetsam_poseidon2b::native::poseidon2b_hash_bytes;
use jetsam_poseidon2b::primitives::{derive_address, Address, SpendSecret};
use jetsam_tx::{
    output_bitmap_bit, PagedSpendIntent, TxBody, TxInput, TxOutput, TxPage, PAGED_SPEND_END_BIT,
    PAGED_SPEND_START_BIT, TX_INPUTS, TX_OUTPUTS,
};

/// One spendable slot as `getSlotsByOwner` reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Utxo {
    pub slot_index: u32,
    pub amount: u64,
    pub creation_id: u64,
}

const ACTOR_DOMAIN: &[u8] = b"JTM/REHEARSAL/V1.7/ACTOR-TEST-KEY";

/// The spend secret of the rehearsal actor `name` (TEST key).
pub fn actor_secret(name: &str) -> [u8; 32] {
    poseidon2b_hash_bytes(ACTOR_DOMAIN, name.as_bytes())
}

/// The outputs of a committee update's payment, before the change.
pub fn committee_update_outputs(payload: &[u8]) -> Vec<(Address, u64)> {
    vec![(committee_update_marker(payload), 0)]
}

/// The outputs of a deposit of `amount` for the L1 `recipient`, before the
/// change: the definition `jetsam-cli bridge deposit` builds with too.
pub use jetsam_rpc::bridge::deposit_outputs;

/// The outputs of a guardian object's payment, before the change.
pub fn guardian_outputs(attestation: &GuardianAttestation) -> Vec<(Address, u64)> {
    vec![(attestation.object().marker(), 0)]
}

/// Largest-first selection of `utxos` (minus `reserved` slots, minus
/// coinbase slots when `spendable_coinbase` is false) covering `need`:
/// the selection and its change.
pub fn select(
    utxos: &[Utxo],
    reserved: &BTreeSet<u32>,
    need: u64,
    spendable_coinbase: bool,
) -> anyhow::Result<(Vec<Utxo>, u64)> {
    let mut candidates: Vec<Utxo> = utxos
        .iter()
        .copied()
        .filter(|u| !reserved.contains(&u.slot_index))
        .filter(|u| spendable_coinbase || !is_coinbase_creation_id(u.creation_id))
        .collect();
    candidates.sort_by(|a, b| {
        b.amount
            .cmp(&a.amount)
            .then(a.slot_index.cmp(&b.slot_index))
    });
    let mut picked = Vec::new();
    let mut total = 0u64;
    for u in candidates {
        if total >= need {
            break;
        }
        total = total.checked_add(u.amount).context("amount overflow")?;
        picked.push(u);
    }
    ensure!(
        total >= need,
        "not enough: need {need} µ, {total} µ spendable"
    );
    Ok((picked, total - need))
}

/// The pages of one logical transaction from `input_owner`: `inputs` eight
/// per page, `outputs` two per page (each on the slot hint of its rank), the
/// fee on the first page.
pub fn pack_group(
    input_owner: Address,
    inputs: &[Utxo],
    outputs: &[(Address, u64)],
    fee: u64,
    epoch_anchor: [u8; 32],
    slot_hints: &[u32],
) -> anyhow::Result<Vec<TxPage>> {
    ensure!(
        slot_hints.len() >= outputs.len(),
        "{} outputs, {} slot hints",
        outputs.len(),
        slot_hints.len()
    );
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
            .map_err(|e| anyhow::anyhow!("page: {e:?}"))?,
        );
    }
    Ok(pages)
}

/// The release group of a burn from the vault slots `inputs`: the owner
/// receives `amount − R`, then the marker, then the change back to the
/// vault; its fee is `R` (D59). Returns the pages and `R`.
pub fn release_group(
    release: &VaultRelease,
    inputs: &[Utxo],
    epoch_anchor: [u8; 32],
    slot_hints: &[u32],
) -> anyhow::Result<(Vec<TxPage>, u64)> {
    let total = inputs.iter().map(|u| u.amount).sum::<u64>();
    ensure!(
        total >= release.amount,
        "the vault slots hold {total} µ, the release takes {} µ",
        release.amount
    );
    let change = total - release.amount;
    let outputs_count = 2 + u64::from(change > 0);
    let r = VaultRules::current().release_fee_for(inputs.len() as u64, outputs_count);
    let paid = release
        .amount
        .checked_sub(r)
        .context("a release of less than R")?;
    let mut outputs = vec![(release.owner, paid), (release.marker(), 0)];
    if change > 0 {
        outputs.push((VAULT_ADDRESS, change));
    }
    let pages = pack_group(VAULT_ADDRESS, inputs, &outputs, r, epoch_anchor, slot_hints)?;
    Ok((pages, r))
}

/// The block ids and the chain work (genesis included, little-endian as
/// the node stores it) of `headers`, the branch's headers from genesis.
pub fn branch_of(headers: &[BlockHeader]) -> (Vec<[u8; 32]>, Vec<[u8; 32]>) {
    let ids = headers.iter().map(block_id).collect();
    let mut sum = [0u8; 32];
    let work = headers
        .iter()
        .map(|h| {
            sum = add_work(&sum, &block_work(&h.difficulty_target));
            sum
        })
        .collect();
    (ids, work)
}

/// The rehearsal attestation of height `x` of the branch `headers`.
pub fn attestation_of(headers: &[BlockHeader], x: u64) -> anyhow::Result<GuardianAttestation> {
    ensure!(x >= 7, "an attestation describes a height of at least 7");
    ensure!(
        (x as usize) < headers.len(),
        "height {x} above the {} headers given",
        headers.len()
    );
    for (h, header) in headers.iter().enumerate() {
        ensure!(
            header.height == h as u64,
            "header {h} is at height {}",
            header.height
        );
    }
    let (ids, work) = branch_of(&headers[..=x as usize]);
    let eight: [[u8; 32]; 8] = ids[x as usize - 7..=x as usize]
        .try_into()
        .expect("eight ids");
    Ok(rehearsal_attestation(x, &eight, &work[x as usize]))
}

// ---------------------------------------------------------------------------
// The command line and the node's RPC
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "jetsam-rehearsal",
    about = "PRIVATE rehearsal tool (rehearsal-v1-7 builds only): synthetic L1 objects and \
             test-key attestations, handed to a rehearsal node"
)]
struct Cli {
    /// The rehearsal node's JSON-RPC endpoint.
    #[arg(long, default_value = "http://127.0.0.1:19511")]
    rpc: String,
    /// Slots spent by intents not yet mined, one per line (read, then appended to).
    #[arg(long)]
    reserve: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// The address of a rehearsal actor (TEST key).
    Actor {
        #[arg(long)]
        name: String,
    },
    /// The synthetic L1 state of a burns file.
    L1State {
        #[arg(long)]
        burns: PathBuf,
    },
    /// The anchor update (`--anchor`) or the rotation out of `--period`, paid by an actor.
    CommitteeUpdate {
        #[arg(long)]
        actor: String,
        #[arg(long)]
        anchor: bool,
        #[arg(long)]
        period: Option<u64>,
    },
    /// A deposit to the vault for an L1 recipient (20 bytes hex), paid by an actor.
    Deposit {
        #[arg(long)]
        actor: String,
        #[arg(long)]
        amount_micro: u64,
        #[arg(long)]
        recipient: String,
    },
    /// The release of burn `n` of a burns file under a finality update of `period`.
    Release {
        #[arg(long)]
        burns: PathBuf,
        #[arg(long)]
        n: u64,
        #[arg(long)]
        period: u64,
        #[arg(long, default_value_t = 1_000)]
        block_number: u64,
    },
    /// The test-key attestation of height `height` of the node's branch, paid by an actor.
    Attest {
        #[arg(long)]
        actor: String,
        #[arg(long)]
        height: u64,
    },
    /// The attestation body (hex) of height `height`, for `walletBuildGuardianPayment`.
    AttestBody {
        #[arg(long)]
        height: u64,
    },
}

struct Node {
    client: reqwest::Client,
    url: String,
}

impl Node {
    async fn call(&self, method: &str, params: Value) -> anyhow::Result<Value> {
        let body = json!({"jsonrpc": "2.0", "id": 1, "method": format!("jetsam_{method}"), "params": params});
        let reply: Value = self
            .client
            .post(&self.url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("{method}: {}", self.url))?
            .json()
            .await
            .with_context(|| format!("{method}: JSON"))?;
        if let Some(error) = reply.get("error") {
            bail!("{method}: {error}");
        }
        Ok(reply["result"].clone())
    }

    async fn utxos(&self, owner: &Address) -> anyhow::Result<Vec<Utxo>> {
        let slots = self
            .call("getSlotsByOwner", json!([owner.to_bech32()]))
            .await?;
        Ok(slots
            .as_array()
            .context("getSlotsByOwner: not a list")?
            .iter()
            .filter(|s| s["empty"].as_bool() != Some(true))
            .filter_map(|s| {
                Some(Utxo {
                    slot_index: u32::try_from(s["slot_index"].as_u64()?).ok()?,
                    amount: s["value"].as_u64()?,
                    creation_id: s["creation_id"].as_u64()?,
                })
            })
            .collect())
    }

    async fn anchor(&self) -> anyhow::Result<[u8; 32]> {
        let text = self.call("getEpochAnchor", json!([])).await?;
        let bytes = hex::decode(text.as_str().context("getEpochAnchor: not a string")?)?;
        bytes
            .try_into()
            .map_err(|_| anyhow::anyhow!("getEpochAnchor: not 32 bytes"))
    }

    async fn hints(&self, count: usize) -> anyhow::Result<Vec<u32>> {
        let hints = self.call("getSlotHints", json!([count])).await?;
        Ok(serde_json::from_value(hints)?)
    }

    async fn fee(&self, inputs: usize, outputs: usize) -> anyhow::Result<u64> {
        let estimate = self
            .call("estimateFeeDetailed", json!([inputs, outputs]))
            .await?;
        estimate["fee_micro_jtm"]
            .as_u64()
            .context("estimateFeeDetailed: no fee")
    }

    async fn headers(&self, to: u64) -> anyhow::Result<Vec<BlockHeader>> {
        let mut out = Vec::with_capacity(to as usize + 1);
        for h in 0..=to {
            let text = self.call("getHeaderByHeight", json!([h])).await?;
            let bytes = hex::decode(text.as_str().with_context(|| format!("no header {h}"))?)?;
            out.push(
                BlockHeader::from_bytes(&bytes)
                    .map_err(|e| anyhow::anyhow!("header {h}: {e:?}"))?,
            );
        }
        Ok(out)
    }
}

fn read_reserved(path: Option<&Path>) -> anyhow::Result<BTreeSet<u32>> {
    let Some(path) = path else {
        return Ok(BTreeSet::new());
    };
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    std::fs::read_to_string(path)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse::<u32>().map_err(Into::into))
        .collect()
}

fn append_reserved(path: Option<&Path>, inputs: &[Utxo]) -> anyhow::Result<()> {
    use std::io::Write;
    let Some(path) = path else {
        return Ok(());
    };
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    for u in inputs {
        writeln!(file, "{}", u.slot_index)?;
    }
    Ok(())
}

fn prove(pages: Vec<TxPage>, secret: [u8; 32]) -> anyhow::Result<PagedSpendIntent> {
    let bundle = prove_paged_spend_authorization(
        &pages,
        OwnerAuthWitness::new(SpendSecret::from_bytes(secret)),
    )
    .map_err(|e| anyhow::anyhow!("authorization: {e}"))?;
    let bytes = bundle
        .to_bytes()
        .map_err(|e| anyhow::anyhow!("authorization bytes: {e:?}"))?;
    PagedSpendIntent::new(pages, bytes).map_err(|e| anyhow::anyhow!("intent: {e:?}"))
}

/// An actor's payment of `outputs` (markers, deposit), its change back to it.
async fn actor_payment(
    node: &Node,
    reserve: Option<&Path>,
    actor: &str,
    outputs: Vec<(Address, u64)>,
) -> anyhow::Result<(PagedSpendIntent, Vec<Utxo>, u64)> {
    let secret = actor_secret(actor);
    let owner = derive_address(&SpendSecret::from_bytes(secret));
    let utxos = node.utxos(&owner).await?;
    let reserved = read_reserved(reserve)?;
    let paid: u64 = outputs.iter().map(|(_, a)| a).sum();
    // The fee of the shape the selection makes (at least: a fee above the
    // minimum is valid).
    let mut fee = node.fee(1, outputs.len() + 1).await?;
    let (inputs, change) = loop {
        let (inputs, change) = select(&utxos, &reserved, paid + fee, true)
            .with_context(|| format!("actor {actor} ({owner})"))?;
        let needed = node
            .fee(inputs.len(), outputs.len() + usize::from(change > 0))
            .await?;
        if needed <= fee {
            break (inputs, change);
        }
        fee = needed;
    };
    let mut all = outputs;
    if change > 0 {
        all.push((owner, change));
    }
    let hints = node.hints(all.len()).await?;
    let pages = pack_group(owner, &inputs, &all, fee, node.anchor().await?, &hints)?;
    let intent = prove(pages, secret)?;
    append_reserved(reserve, &inputs)?;
    Ok((intent, inputs, fee))
}

fn read_burns(path: &Path) -> anyhow::Result<Vec<Burn>> {
    let list: Vec<Value> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    list.iter()
        .map(|b| {
            let owner_text = b["owner"].as_str().context("burn: owner")?;
            let owner = match Address::parse(owner_text) {
                Ok(a) => a.0,
                Err(_) => hex::decode(owner_text)?
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("owner: 32 bytes"))?,
            };
            Ok(Burn {
                n: b["n"].as_u64().context("burn: n")?,
                owner,
                amount: b["amount"].as_u64().context("burn: amount")?,
            })
        })
        .collect()
}

async fn run(cli: Cli) -> anyhow::Result<Value> {
    let node = Node {
        client: reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1_200))
            .build()?,
        url: cli.rpc.clone(),
    };
    let reserve = cli.reserve.as_deref();
    Ok(match cli.command {
        Command::Actor { name } => {
            let address = derive_address(&SpendSecret::from_bytes(actor_secret(&name)));
            json!({"actor": name, "address": address.to_bech32(), "hex": hex::encode(address.0)})
        }
        Command::L1State { burns } => {
            let burns = read_burns(&burns)?;
            let state = bridge_state(&burns);
            json!({"state_root": hex::encode(state.state_root), "burns": burns.len()})
        }
        Command::CommitteeUpdate {
            actor,
            anchor,
            period,
        } => {
            let update = match (anchor, period) {
                (true, None) => anchor_update(),
                (false, Some(p)) => committee_update(p),
                _ => bail!("--anchor or --period <P>"),
            };
            let payload = update.encode_payload();
            let (payment, inputs, fee) =
                actor_payment(&node, reserve, &actor, committee_update_outputs(&payload)).await?;
            let txid = hex::encode(payment.logical_txid().0);
            let intent =
                DestinationIntent::CommitteeUpdate(CommitteeUpdateIntent { payload, payment })
                    .encode()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
            let admitted = node
                .call("submitDestinationObject", json!([hex::encode(intent)]))
                .await?;
            json!({"admitted": admitted, "txid": txid, "fee": fee, "inputs": inputs.len()})
        }
        Command::Deposit {
            actor,
            amount_micro,
            recipient,
        } => {
            let recipient: [u8; 20] = hex::decode(recipient.trim_start_matches("0x"))?
                .try_into()
                .map_err(|_| anyhow::anyhow!("recipient: 20 bytes"))?;
            let (payment, inputs, fee) = actor_payment(
                &node,
                reserve,
                &actor,
                deposit_outputs(amount_micro, &recipient),
            )
            .await?;
            let txid = hex::encode(payment.logical_txid().0);
            let bytes = payment.to_bytes().map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let submitted = node
                .call("submitTxIntent", json!([hex::encode(bytes)]))
                .await?;
            json!({"submitted": submitted, "txid": txid, "fee": fee, "inputs": inputs.len()})
        }
        Command::Release {
            burns,
            n,
            period,
            block_number,
        } => {
            let burns = read_burns(&burns)?;
            let burn = burns
                .iter()
                .find(|b| b.n == n)
                .with_context(|| format!("no burn {n} in the file"))?;
            let l1 = bridge_state(&burns);
            let release = VaultRelease {
                n,
                owner: Address(burn.owner),
                amount: burn.amount,
                storage_proof: l1.storage_proof(n),
            };
            let vault = node.call("getVaultState", json!([])).await?;
            let floor = vault["floor"].as_u64().context("getVaultState: floor")?;
            let candidates: Vec<Utxo> = node
                .utxos(&VAULT_ADDRESS)
                .await?
                .into_iter()
                .filter(|u| u.creation_id > floor && !is_coinbase_creation_id(u.creation_id))
                .collect();
            let (inputs, _) = select(&candidates, &read_reserved(reserve)?, burn.amount, false)
                .context("vault slots for this release")?;
            let hints = node.hints(3).await?;
            let (pages, r) = release_group(&release, &inputs, node.anchor().await?, &hints)?;
            let payment = prove(pages, VAULT_SECRET)?;
            let txid = hex::encode(payment.logical_txid().0);
            // The account proof and, since A5-8, the genesis proof after it.
            let account_proof = l1.account_payload();
            let intent = DestinationIntent::Release(ReleaseIntent {
                release,
                finality: finality_update(period, l1.state_root, block_number).encode_payload(),
                account_proof,
                payment,
                base: None,
                fast: None,
            })
            .encode()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            let admitted = node
                .call("submitDestinationObject", json!([hex::encode(intent)]))
                .await?;
            append_reserved(reserve, &inputs)?;
            json!({"admitted": admitted, "txid": txid, "release_fee": r, "vault_inputs": inputs.len(),
                   "l1_state_root": hex::encode(l1.state_root)})
        }
        Command::Attest { actor, height } => {
            let headers = node.headers(height).await?;
            let attestation = attestation_of(&headers, height)?;
            let (payment, inputs, fee) =
                actor_payment(&node, reserve, &actor, guardian_outputs(&attestation)).await?;
            let txid = hex::encode(payment.logical_txid().0);
            let intent = GuardianIntent {
                object: attestation.object(),
                payment,
            }
            .encode()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
            let admitted = node
                .call("submitGuardianObject", json!([hex::encode(intent)]))
                .await?;
            json!({"admitted": admitted, "txid": txid, "fee": fee, "inputs": inputs.len(),
                   "attested_height": height, "block_id": hex::encode(block_id(&headers[height as usize]))})
        }
        Command::AttestBody { height } => {
            let headers = node.headers(height).await?;
            let attestation = attestation_of(&headers, height)?;
            json!({"height": height, "body": hex::encode(attestation.body())})
        }
    })
}

#[tokio::main]
async fn main() {
    match run(Cli::parse()).await {
        Ok(value) => println!("{value}"),
        Err(error) => {
            eprintln!("jetsam-rehearsal: {error:#}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
#[path = "jetsam_rehearsal/tests.rs"]
mod tests;
