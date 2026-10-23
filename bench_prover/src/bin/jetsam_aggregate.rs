// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jetsam-aggregate`: the aggregator of the v1.6 batch slot, off
//! the node. It takes client proofs of registered matrices, builds one batch
//! proof of them (a chain of batch steps, then the junction on the parent's
//! lanes), and decides a batch proof the way a miner does before carrying it.
//!
//! ```text
//! jetsam-aggregate class
//! jetsam-aggregate build  (--registry D[,D...] --base | --rpc HOST:PORT) --matrix FILE... --client D=FILE... --out FILE
//! jetsam-aggregate decide (--registry D[,D...] --base | --rpc HOST:PORT) --matrix FILE... --batch FILE
//! jetsam-aggregate submit --rpc HOST:PORT --matrix FILE... --client D=FILE... [--fee MICRO_JTM] [--out FILE]
//! ```
//!
//! - `class` freezes the batch class over the hiding client form (`canonical_z`, m = 22)
//!   and prints `D_batch`, its post-commit class, its rows and the wire bound
//!   of its proofs.
//! - `--registry` is the chain's registry, `D` in entry order (hex);
//!   `--matrix` the registered matrix files (`jetsam_client_demo`'s
//!   `matrix.bin`); `--client D=FILE` a client proof (`proof-N.bin`) of the
//!   matrix `D`, in batch order.
//! - `--base`: the batch is a junction on dead parent lanes — the first
//!   block after the v1.6 height, or a parent whose chain carried no batch
//!   yet.
//! - `--rpc HOST:PORT`: the registry leaves and the lanes the node's tip
//!   publishes are read from the node (`jetsam_getClientBatchParent`):
//!   the batch is a junction on them, for the block after that tip.
//! - `submit` builds the batch over the node's tip, has the node's wallet
//!   build and prove its payment (`jetsam_walletBuildClientPayment`), the
//!   node assemble the bundle (`jetsam_buildClientBatchBundle`) and receive
//!   it (`jetsam_submitClientBatch`: decided against the tip, held for the
//!   miners, announced to peers).
//!
//! The batch class is the chain's (`CLIENT_BATCH_CLASS`), frozen at each run
//! (tens of seconds) and checked against the constant. **Test network
//! only**: no v1.6 height is armed on either profile.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use jetsam_ivc_core::field_r1cs::FieldR1cs;
use jetsam_recursive::{
    aggregate_client_batch, decide_client_batch_bytes, AggregationClass, AggregationClient,
    ParentLanes,
};

const USAGE: &str = "usage:
  jetsam-aggregate class
  jetsam-aggregate build  (--registry D[,D...] --base | --rpc HOST:PORT) --matrix FILE... --client D=FILE... --out FILE
  jetsam-aggregate decide (--registry D[,D...] --base | --rpc HOST:PORT) --matrix FILE... --batch FILE
  jetsam-aggregate submit --rpc HOST:PORT --matrix FILE... --client D=FILE... [--fee MICRO_JTM] [--out FILE]";

/// The largest matrix file a node fetches (the consensus bound).
const MATRIX_MAX_BYTES: usize =
    jetsam_chain::consensus::client_objects::CLIENT_MATRIX_MAX_FILE_BYTES as usize;

#[derive(Default)]
struct Args {
    registry: Vec<[u8; 32]>,
    matrices: Vec<PathBuf>,
    clients: Vec<([u8; 32], PathBuf)>,
    batch: Option<PathBuf>,
    out: Option<PathBuf>,
    base: bool,
    rpc: Option<String>,
    fee: Option<u64>,
}

fn hex32(bytes: &[u8; 32]) -> String {
    hex::encode(bytes)
}

fn parse_digest(text: &str) -> Result<[u8; 32], String> {
    let bytes = hex::decode(text.trim()).map_err(|error| format!("{text}: {error}"))?;
    bytes
        .try_into()
        .map_err(|_| format!("{text}: a digest is 32 bytes"))
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut parsed = Args::default();
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{flag} takes a value"));
        match flag.as_str() {
            "--registry" => {
                for digest in value()?.split(',').filter(|text| !text.is_empty()) {
                    parsed.registry.push(parse_digest(digest)?);
                }
            }
            "--matrix" => parsed.matrices.push(PathBuf::from(value()?)),
            "--client" => {
                let value = value()?;
                let (digest, path) = value
                    .split_once('=')
                    .ok_or_else(|| format!("--client takes D=FILE, not {value}"))?;
                parsed.clients.push((parse_digest(digest)?, PathBuf::from(path)));
            }
            "--batch" => parsed.batch = Some(PathBuf::from(value()?)),
            "--out" => parsed.out = Some(PathBuf::from(value()?)),
            "--base" => parsed.base = true,
            "--rpc" => parsed.rpc = Some(value()?),
            "--fee" => {
                parsed.fee = Some(value()?.parse().map_err(|error| format!("--fee: {error}"))?)
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    Ok(parsed)
}

fn freeze() -> Result<&'static AggregationClass, String> {
    let started = Instant::now();
    let class = AggregationClass::chain()?;
    eprintln!(
        "batch class frozen in {:.1} s: m={} rows={} D_batch={}",
        started.elapsed().as_secs_f64(),
        class.m(),
        class.useful_rows(),
        hex32(&class.digest())
    );
    Ok(class)
}

/// The chain's registry, padded to its capacity.
fn leaves(class: &AggregationClass, registry: &[[u8; 32]]) -> Result<Vec<[u8; 32]>, String> {
    let capacity = class.form().registry_capacity();
    if registry.is_empty() || registry.len() > capacity {
        return Err(format!("--registry names 1 to {capacity} digests"));
    }
    let mut leaves = registry.to_vec();
    leaves.resize(capacity, [0u8; 32]);
    Ok(leaves)
}

/// The matrix files, each decoded under the client form and keyed by its
/// structural digest `D`, which must be in the registry.
fn matrices(
    class: &AggregationClass,
    files: &[PathBuf],
    registry: &[[u8; 32]],
) -> Result<BTreeMap<[u8; 32], Arc<FieldR1cs>>, String> {
    let mut matrices = BTreeMap::new();
    for path in files {
        let mut file = std::fs::File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
        let matrix = FieldR1cs::read_artifact_unbound(&mut file, class.form().shape(), MATRIX_MAX_BYTES)
            .map_err(|error| format!("{}: not a matrix of the client form: {error:?}", path.display()))?;
        let digest = matrix.structural_statement_digest();
        if !registry.contains(&digest) {
            return Err(format!("{}: D {} is not in --registry", path.display(), hex32(&digest)));
        }
        matrices.insert(digest, Arc::new(matrix));
    }
    Ok(matrices)
}

/// One JSON-RPC call to the node at `address` (`HOST:PORT`, plain HTTP).
fn rpc_call(address: &str, method: &str, params: serde_json::Value) -> Result<serde_json::Value, String> {
    use std::io::{Read as _, Write as _};
    let body = serde_json::json!({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).to_string();
    let mut stream =
        std::net::TcpStream::connect(address).map_err(|error| format!("{address}: {error}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(900)))
        .map_err(|error| error.to_string())?;
    write!(
        stream,
        "POST / HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .map_err(|error| format!("{address}: {error}"))?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response).map_err(|error| format!("{address}: {error}"))?;
    let response = String::from_utf8_lossy(&response);
    let (_, json) = response
        .split_once("\r\n\r\n")
        .ok_or_else(|| format!("{method}: no HTTP body"))?;
    let value: serde_json::Value =
        serde_json::from_str(json.trim()).map_err(|error| format!("{method}: {error}: {json}"))?;
    if let Some(error) = value.get("error") {
        return Err(format!("{method}: {error}"));
    }
    value.get("result").cloned().ok_or_else(|| format!("{method}: no result"))
}

/// What a batch is built over: the registry leaves and the parent lanes —
/// dead (`--base` with `--registry`) or the node's tip's (`--rpc`).
fn parent_of(class: &AggregationClass, args: &Args) -> Result<(Vec<[u8; 32]>, ParentLanes), String> {
    match (&args.rpc, args.base) {
        (Some(address), false) => {
            let parent = rpc_call(address, "jetsam_getClientBatchParent", serde_json::json!([]))?;
            let text = |key: &str| {
                parent
                    .get(key)
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| format!("jetsam_getClientBatchParent: no {key}"))
            };
            if parse_digest(text("batch_class")?)? != class.digest() {
                return Err("the node's batch class is not this tool's".into());
            }
            let leaves = parent
                .get("leaves")
                .and_then(|value| value.as_array())
                .ok_or("jetsam_getClientBatchParent: no leaves")?
                .iter()
                .map(|leaf| parse_digest(leaf.as_str().unwrap_or_default()))
                .collect::<Result<Vec<_>, _>>()?;
            let lanes_bytes = hex::decode(text("lanes_hex")?).map_err(|error| error.to_string())?;
            let lanes =
                ParentLanes::decode(&lanes_bytes, class.form().registry_capacity(), class.form().shape().k_log)?;
            eprintln!(
                "parent: tip {} ({}), {} live entry lane(s), D_batch lane {}",
                parent.get("tip_height").and_then(|value| value.as_u64()).unwrap_or_default(),
                text("tip_hash")?,
                lanes.entries.iter().filter(|lane| lane.is_some()).count(),
                if lanes.batch.is_some() { "live" } else { "dead" }
            );
            Ok((leaves, lanes))
        }
        (None, true) => Ok((
            leaves(class, &args.registry)?,
            ParentLanes::empty(class.form().registry_capacity()),
        )),
        _ => Err(format!("give either --registry ... --base, or --rpc HOST:PORT\n{USAGE}")),
    }
}

/// The non-empty registry leaves (the `D`s a matrix may be of).
fn registered(leaves: &[[u8; 32]]) -> Vec<[u8; 32]> {
    leaves.iter().copied().filter(|leaf| *leaf != [0u8; 32]).collect()
}

fn run() -> Result<(), String> {
    let mut argv = std::env::args().skip(1);
    let command = argv.next().ok_or_else(|| USAGE.to_owned())?;
    let args = parse_args(argv)?;
    match command.as_str() {
        "class" => {
            let class = freeze()?;
            println!("D_batch          {}", hex32(&class.digest()));
            println!("post-commit      {}", hex32(&class.post_commit()));
            println!("m                {}", class.m());
            println!("rows             {}", class.useful_rows());
            println!("max wire bytes   {}", class.max_wire_bytes()?);
        }
        "build" | "submit" => {
            if args.clients.is_empty() {
                return Err(format!("{command} takes at least one --client\n{USAGE}"));
            }
            if command == "submit" && args.rpc.is_none() {
                return Err(format!("submit takes --rpc\n{USAGE}"));
            }
            let out = match (&args.out, command.as_str()) {
                (Some(out), _) => out.clone(),
                (None, "submit") => std::env::temp_dir().join("jetsam-aggregate-batch.bin"),
                (None, _) => return Err("build takes --out".into()),
            };
            let class = freeze()?;
            let (leaves, parent) = parent_of(class, &args)?;
            let matrices = matrices(class, &args.matrices, &registered(&leaves))?;
            // Each client proof with the matrix of the `D` it names.
            let clients = args
                .clients
                .iter()
                .map(|(digest, path)| {
                    let proof_bytes =
                        std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
                    let matrix = matrices
                        .get(digest)
                        .ok_or_else(|| format!("{}: no --matrix of D {}", path.display(), hex32(digest)))?;
                    Ok(AggregationClient {
                        proof_bytes,
                        matrix: Arc::clone(matrix),
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let started = Instant::now();
            let batch = aggregate_client_batch(class, &leaves, &clients, &parent, &matrices)?;
            std::fs::write(&out, &batch.proof_bytes).map_err(|error| format!("{}: {error}", out.display()))?;
            eprintln!(
                "batch of {} built in {:.1} s ({} steps: {:?} s), {} bytes (bound {})",
                batch.statement.count,
                started.elapsed().as_secs_f64(),
                batch.step_seconds.len(),
                batch.step_seconds.iter().map(|s| (s * 10.0).round() / 10.0).collect::<Vec<_>>(),
                batch.proof_bytes.len(),
                class.max_wire_bytes()?,
            );
            println!("batch_class      {}", hex32(&class.digest()));
            println!("statements_root  {}", hex32(&batch.statement.statements_root));
            println!("count            {}", batch.statement.count);
            println!(
                "entries used     {:?}",
                batch.statement.mask.iter().enumerate().filter(|(_, used)| **used).map(|(index, _)| index).collect::<Vec<_>>()
            );
            println!("proof            {} ({} bytes)", out.display(), batch.proof_bytes.len());
            if command == "submit" {
                let address = args.rpc.as_deref().expect("checked above");
                let count = u32::try_from(batch.statement.count).map_err(|error| error.to_string())?;
                let payment = rpc_call(
                    address,
                    "jetsam_walletBuildClientPayment",
                    serde_json::json!([
                        {
                            "kind": "batch",
                            "batch_class": hex32(&class.digest()),
                            "statements_root": hex32(&batch.statement.statements_root),
                            "count": count,
                        },
                        args.fee,
                    ]),
                )?;
                let payment_hex = payment
                    .get("payment_hex")
                    .and_then(|value| value.as_str())
                    .ok_or("jetsam_walletBuildClientPayment: no payment_hex")?;
                println!(
                    "payment          txid {} fee {} µJTM",
                    payment.get("txid").and_then(|value| value.as_str()).unwrap_or_default(),
                    payment.get("fee_micro_jtm").and_then(|value| value.as_u64()).unwrap_or_default()
                );
                let bundle = rpc_call(
                    address,
                    "jetsam_buildClientBatchBundle",
                    serde_json::json!([
                        hex32(&class.digest()),
                        hex32(&batch.statement.statements_root),
                        count,
                        payment_hex,
                        hex::encode(&batch.proof_bytes),
                    ]),
                )?;
                let started = Instant::now();
                let submitted =
                    rpc_call(address, "jetsam_submitClientBatch", serde_json::json!([bundle]))?;
                eprintln!("submitted in {:.1} s", started.elapsed().as_secs_f64());
                println!("submitted        {submitted}");
            }
        }
        "decide" => {
            let path = args.batch.clone().ok_or("decide takes --batch")?;
            let class = freeze()?;
            let (leaves, parent) = parent_of(class, &args)?;
            let matrices = matrices(class, &args.matrices, &registered(&leaves))?;
            let bytes = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
            let started = Instant::now();
            let decided = decide_client_batch_bytes(class, &bytes, &leaves, &parent, &matrices)?;
            eprintln!("decided in {:.2} s", started.elapsed().as_secs_f64());
            println!("statements_root  {}", hex32(&decided.statement.statements_root));
            println!("count            {}", decided.statement.count);
        }
        _ => return Err(USAGE.to_owned()),
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("jetsam-aggregate: {error}");
        std::process::exit(1);
    }
}
