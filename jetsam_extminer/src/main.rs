// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! # jetsam-miner — External Poseidon2b PoW miner for Jetsam.
//!
//! Connects to any `jetsam` full node via JSON-RPC, fetches a block template,
//! searches for a valid PoW nonce using all available CPU cores (rayon), and
//! returns only that nonce to the node-owned template.
//!
//! ## Usage
//!
//! ```bash
//! # Solo (node on localhost, no auth)
//! jetsam-miner --rpc http://127.0.0.1:9701
//!
//! # Pool (remote node with bearer token)
//! jetsam-miner --rpc https://pool.example.com:9701 --key my-secret-token
//!
//! # Limit threads
//! jetsam-miner --rpc http://127.0.0.1:9701 --threads 4
//! ```
//!
//! ## Template protocol
//!
//! `getBlockTemplate("")` returns:
//!   - `template_id`              — opaque, single-use node capability
//!   - `pow_fields_hex`           — 16-field Poseidon2b PoW input
//!   - `nonce_field_index`        — nonce lane (canonical value: 0)
//!   - `difficulty_target_hex`    — 256-bit LE target
//!   - block metadata             — operator display only
//!
//! The miner calls `submitBlock(template_id, nonce_hex)`, where `nonce_hex` is
//! exactly the 16-byte little-endian nonce in lowercase hex. The worker never
//! receives or submits a block body, HistoryStep witness or proof.

use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use jetsam_core::{Block128, TowerField};
use jetsam_poseidon2b::batch::FixedFieldNonceBatch;
use jetsam_poseidon2b::native::domain::TAG_POWHDR;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
#[command(
    name = "jetsam-miner",
    version,
    about = "External Poseidon2b PoW miner for Jetsam",
    long_about = "Fetches block templates from a Jetsam node and mines blocks \
                  using all available CPU cores.\n\n\
                  The node builds the proven template; this worker only does PoW.\n\
                  Coinbase is the node payout address unless the node enables \
                  --allow-custom-coinbase and the worker supplies --coinbase."
)]
struct Cli {
    /// Check production CPU support and exit without connecting to a node.
    #[arg(long, exclusive = true)]
    check_hardware: bool,

    /// JSON-RPC endpoint of the Jetsam node or pool.
    #[arg(long, default_value = "http://127.0.0.1:9701", value_name = "URL")]
    rpc: String,

    /// Bearer token for pool/external RPC access.
    /// Must match the node's --mining-key flag.
    /// Not needed for solo miners using the default 127.0.0.1 binding.
    #[arg(long, value_name = "TOKEN")]
    key: Option<String>,

    /// Number of PoW threads. 0 = every logical CPU visible to the process.
    #[arg(long, default_value_t = 0, value_name = "N")]
    threads: usize,

    /// Your own payout address (bech32m j1...).
    /// Only works when the node is started with --allow-custom-coinbase.
    /// Leave empty to use the node's configured payout address (pool mode).
    #[arg(long, value_name = "ADDRESS", default_value = "")]
    coinbase: String,

    /// Milliseconds to wait before re-fetching a new template after a solve
    /// or stale detection. Lower = more responsive to new blocks.
    #[arg(long, default_value_t = 500, value_name = "MS")]
    poll_ms: u64,

    /// Log level (error | warn | info | debug).
    #[arg(long, default_value = "info", value_name = "LEVEL")]
    log: String,

    /// Benchmark this machine's TowerHash rate for N seconds and exit, without
    /// touching the network. Reports hashes/second at the chosen --threads, so
    /// you can size how many threads each CPU needs to pull its weight.
    #[arg(long, value_name = "SECONDS")]
    bench: Option<u64>,
}

// ---------------------------------------------------------------------------
// RPC types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct BlockTemplateResponse {
    template_id: String,
    pow_fields_hex: String,
    nonce_field_index: usize,
    difficulty_target_hex: String,
    height: u64,
    expires_in_seconds: u64,
    n_txs: usize,
    /// Nonce region this worker owns, assigned by a pool that serves one
    /// template to several machines. A node answering a solo miner does not
    /// send it, and the miner then keeps its own starting point.
    #[serde(default)]
    nonce_prefix: Option<u32>,
}

#[derive(Debug, Serialize)]
struct JsonRpcRequest<'a, P: Serialize> {
    jsonrpc: &'a str,
    id: u32,
    method: &'a str,
    params: P,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse<T> {
    result: Option<T>,
    error: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// RPC client
// ---------------------------------------------------------------------------

struct RpcClient {
    url: String,
    key: Option<String>,
    http: reqwest::blocking::Client,
}

impl RpcClient {
    fn new(url: &str, key: Option<String>) -> Self {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("build HTTP client");
        Self {
            url: url.to_string(),
            key,
            http,
        }
    }

    fn call<P: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        params: P,
    ) -> Result<R> {
        let body = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 1,
            method,
            params,
        };
        let mut req = self.http.post(&self.url).json(&body);
        if let Some(ref token) = self.key {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        let resp = req.send().with_context(|| format!("POST {}", self.url))?;
        let status = resp.status();
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(anyhow!(
                "401 Unauthorized — node requires --key <token>. \
                 Make sure --mining-key matches on the node."
            ));
        }
        if !status.is_success() {
            return Err(anyhow!("HTTP {status} from {}", self.url));
        }
        let rpc: JsonRpcResponse<R> = resp.json().context("decode JSON-RPC response")?;
        if let Some(err) = rpc.error {
            return Err(anyhow!("RPC error: {err}"));
        }
        rpc.result
            .ok_or_else(|| anyhow!("RPC returned null result"))
    }

    fn get_template(&self, coinbase: &str) -> Result<BlockTemplateResponse> {
        self.call("jetsam_getBlockTemplate", [coinbase])
    }

    fn submit_nonce(&self, template_id: &str, nonce: u128) -> Result<String> {
        let nonce_hex = hex::encode(nonce.to_le_bytes());
        self.call("jetsam_submitBlock", (template_id, nonce_hex))
    }
}

// ---------------------------------------------------------------------------
// PoW
// ---------------------------------------------------------------------------

const CHUNK_SIZE: u128 = 10_000_000;
const DIGEST_BATCH: usize = 256;
const POW_HEADER_FIELD_COUNT: usize = 16;
// JETSAM CHANGE: 0, was 10. This constant is DUPLICATED from
// jetsam_chain::consensus::pow so the external miner stays dependency-free;
// the two must move together. The runtime check below is the safety net.
const POW_NONCE_FIELD_INDEX: usize = 0;
const POW_FIELDS_HEX_BYTES: usize = POW_HEADER_FIELD_COUNT * 16;
const TEMPLATE_SUBMIT_MARGIN: Duration = Duration::from_secs(1);
/// A pool-assigned region is the top 32 bits of the 128-bit nonce, leaving
/// 2^96 nonces to each worker — more than any machine can exhaust before the
/// template expires, so two workers never meet.
const NONCE_REGION_SHIFT: u32 = 96;

fn template_search_deadline(received_at: Instant, expires_in_seconds: u64) -> Instant {
    let usable = Duration::from_secs(expires_in_seconds).saturating_sub(TEMPLATE_SUBMIT_MARGIN);
    received_at.checked_add(usable).unwrap_or(received_at)
}

/// Search for a valid nonce using all rayon threads.
/// Returns `Some(nonce)` or `None` when the node-owned template is too close
/// to expiry to submit safely.
///
/// `cursor` is where this worker's search resumes, and it only ever moves
/// forward. A template that expires mid-search is fetched again and searched
/// again; restarting from a fresh point re-hashed the range the previous pass
/// had already covered.
fn search_nonce(
    pow_fields: &[Block128; POW_HEADER_FIELD_COUNT],
    target: &[u8; 32],
    deadline: Instant,
    cursor: &mut u128,
) -> Option<u128> {
    let num_threads = rayon::current_num_threads();
    let per_thread = CHUNK_SIZE.div_ceil(num_threads as u128);

    let mut chunk_start = *cursor;

    loop {
        if Instant::now() >= deadline {
            *cursor = chunk_start;
            return None;
        }

        let chunk_end = chunk_start + CHUNK_SIZE;
        let solution = (0..num_threads).into_par_iter().find_map_any(|tid| {
            let ts = chunk_start + tid as u128 * per_thread;
            let te = (ts + per_thread).min(chunk_end);
            if ts >= te {
                return None;
            }

            let fields = *pow_fields;
            let mut hasher = FixedFieldNonceBatch::new(TAG_POWHDR, &fields, POW_NONCE_FIELD_INDEX);
            let mut digests = [[0u8; 32]; DIGEST_BATCH];
            let mut nonce = ts;
            while nonce < te {
                if Instant::now() >= deadline {
                    return None;
                }
                let n = ((te - nonce).min(DIGEST_BATCH as u128)) as usize;
                hasher.hash_into(nonce, &mut digests[..n]);
                for (i, hash) in digests[..n].iter().enumerate() {
                    if le256_lt(hash, target) {
                        return Some(nonce + i as u128);
                    }
                }
                nonce += n as u128;
            }
            None
        });

        chunk_start = chunk_start.wrapping_add(CHUNK_SIZE);
        if solution.is_some() {
            *cursor = chunk_start;
            return solution;
        }
    }
}

fn decode_pow_fields_hex(hex_str: &str) -> Result<[Block128; POW_HEADER_FIELD_COUNT]> {
    let bytes = hex::decode(hex_str)?;
    if bytes.len() != POW_FIELDS_HEX_BYTES {
        return Err(anyhow!(
            "pow_fields_hex must be {POW_FIELDS_HEX_BYTES} bytes, got {}",
            bytes.len()
        ));
    }
    let mut fields = [Block128::ZERO; POW_HEADER_FIELD_COUNT];
    for (i, chunk) in bytes.chunks_exact(16).enumerate() {
        fields[i] = Block128::from(u128::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(fields)
}

/// Compare two 32-byte values as 256-bit LE integers: `a < b`.
#[inline]
fn le256_lt(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Mining loop
// ---------------------------------------------------------------------------

fn mine(cli: &Cli) -> Result<()> {
    let rpc = RpcClient::new(&cli.rpc, cli.key.clone());

    // Configure rayon thread pool.
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .context("configure PoW thread pool")?;
    }
    let threads = rayon::current_num_threads();

    eprintln!(
        "jetsam-miner  rpc={}  threads={}  backend={}  poll={}ms",
        cli.rpc,
        threads,
        jetsam_core::cpu::selected_backend(),
        cli.poll_ms,
    );
    if cli.key.is_some() {
        eprintln!("auth: bearer token configured");
    }
    if !cli.coinbase.is_empty() {
        eprintln!(
            "coinbase: {} (custom — node must have --allow-custom-coinbase)",
            cli.coinbase
        );
    } else {
        eprintln!("coinbase: node's payout address (pool mode)");
    }
    eprintln!("Connecting to node...\n");

    let mut blocks_found: u64 = 0;
    let mut last_height: u64 = 0;

    let clock_entropy = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u128;

    // Where this worker's nonce search resumes. Until a pool assigns a region,
    // the start is drawn from the sub-second clock, as before.
    let mut cursor: u128 = clock_entropy & 0xFFFF_FFFF_FFFF_FFFF;
    let mut cursor_region: Option<u32> = None;

    // Where inside an assigned region this process starts.
    //
    // A pool derives the region from the client address, so a second miner on
    // this machine — or another machine behind the same NAT — is handed the
    // same one. Both would then start at the region's first nonce and advance
    // by the same step, hashing identical work forever. This offset separates
    // them. It occupies bits 64..96, so it can never reach the neighbouring
    // region: 2^64 nonces remain above the highest offset.
    let region_offset: u128 = {
        let pid = u128::from(std::process::id());
        ((clock_entropy ^ (pid << 12)) & 0xFFFF_FFFF) << 64
    };

    loop {
        // Fetch template.
        let (tmpl, template_received_at) = match rpc.get_template(&cli.coinbase) {
            Ok(t) => (t, Instant::now()),
            Err(e) => {
                eprintln!("template fetch failed: {e}  — retrying in 2s");
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
        };

        // One template, several machines: each is given a region of the nonce
        // space to itself. Without it every worker starts somewhere in
        // [0, 10^9) and spends most of its time re-hashing what its neighbours
        // already covered.
        if tmpl.nonce_prefix != cursor_region {
            if let Some(prefix) = tmpl.nonce_prefix {
                cursor = (u128::from(prefix) << NONCE_REGION_SHIFT) | region_offset;
                eprintln!("nonce region {prefix} assigned by the pool");
            }
            cursor_region = tmpl.nonce_prefix;
        }

        // Skip if height unchanged and we already solved it.
        if tmpl.height == last_height {
            std::thread::sleep(Duration::from_millis(cli.poll_ms));
            continue;
        }

        let height = tmpl.height;
        let n_txs = tmpl.n_txs;

        let pow_fields =
            decode_pow_fields_hex(&tmpl.pow_fields_hex).context("decode pow_fields_hex")?;
        if tmpl.nonce_field_index != POW_NONCE_FIELD_INDEX {
            return Err(anyhow!(
                "template nonce_field_index must be {POW_NONCE_FIELD_INDEX}, got {}",
                tmpl.nonce_field_index
            ));
        }

        let target: [u8; 32] = hex::decode(&tmpl.difficulty_target_hex)
            .context("decode difficulty_target_hex")?
            .try_into()
            .map_err(|_| anyhow!("difficulty_target must be 32 bytes"))?;

        // Count leading zero bits for display.
        let diff_bits = {
            let mut z = 0u32;
            for i in (0..32usize).rev() {
                if target[i] == 0 {
                    z += 8;
                } else if z.is_multiple_of(8) {
                    z += target[i].leading_zeros();
                    break;
                } else {
                    break;
                }
            }
            z
        };

        eprintln!(
            "┌─ h={height} txs={n_txs} expires={}s diff={diff_bits} leading-zero-bits  \
             target={}…",
            tmpl.expires_in_seconds,
            &tmpl.difficulty_target_hex[tmpl.difficulty_target_hex.len().saturating_sub(8)..]
        );

        let t0 = Instant::now();
        let deadline = template_search_deadline(template_received_at, tmpl.expires_in_seconds);

        let nonce = match search_nonce(&pow_fields, &target, deadline, &mut cursor) {
            Some(n) => n,
            None => {
                eprintln!("└─ EXPIRED  refreshing node-owned template");
                continue;
            }
        };

        let elapsed = t0.elapsed();
        if Instant::now() >= deadline {
            eprintln!("└─ EXPIRED  solution arrived too late; refreshing template");
            continue;
        }

        // Submit only the nonce for this single-use node-owned template.
        match rpc.submit_nonce(&tmpl.template_id, nonce) {
            Ok(hash) => {
                blocks_found += 1;
                last_height = height;
                eprintln!(
                    "└─ SOLVED  nonce={nonce}  time={:.2}s  hash={}…  \
                     [total={blocks_found}]",
                    elapsed.as_secs_f64(),
                    &hash[..20.min(hash.len())],
                );
            }
            Err(e) => {
                let err = e.to_string();
                if err.to_ascii_lowercase().contains("stale") {
                    eprintln!("└─ STALE  template parent lost race; fetching fresh template");
                } else {
                    eprintln!("└─ submit failed: {err}");
                }
            }
        }

        std::thread::sleep(Duration::from_millis(cli.poll_ms));
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run_bench(cli: &Cli) {
    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .expect("thread pool");
    }
    let threads = rayon::current_num_threads();
    let secs = cli.bench.unwrap_or(10).max(1);
    // A fixed dummy header; the hash rate does not depend on its contents.
    let fields = [Block128::from(0x9e37_79b9_7f4a_7c15u128); POW_HEADER_FIELD_COUNT];
    let deadline = Instant::now() + Duration::from_secs(secs);
    eprintln!("benchmarking TowerHash on {threads} thread(s) for {secs}s...");
    let counts: Vec<u128> = (0..threads)
        .into_par_iter()
        .map(|tid| {
            let mut hasher =
                FixedFieldNonceBatch::new(TAG_POWHDR, &fields, POW_NONCE_FIELD_INDEX);
            let mut digests = [[0u8; 32]; DIGEST_BATCH];
            let mut nonce = (tid as u128) << 96;
            let mut done: u128 = 0;
            while Instant::now() < deadline {
                hasher.hash_into(nonce, &mut digests);
                nonce = nonce.wrapping_add(DIGEST_BATCH as u128);
                done += DIGEST_BATCH as u128;
            }
            done
        })
        .collect();
    let total: u128 = counts.iter().sum();
    let hps = (total as f64) / (secs as f64);
    let unit = if hps >= 1.0e6 {
        format!("{:.2} MH/s", hps / 1.0e6)
    } else {
        format!("{:.1} kH/s", hps / 1.0e3)
    };
    println!("TowerHash rate: {unit}  ({threads} threads, {:.1} kH/s per thread)",
             hps / 1.0e3 / threads as f64);
}

fn main() {
    let cli = Cli::parse();
    if cli.check_hardware {
        let report = jetsam_core::cpu::ProductionHardwareReport::detect();
        print!("{report}");
        if report.ready() {
            return;
        }
        let _ = std::io::Write::flush(&mut std::io::stdout());
        std::process::exit(1);
    }
    if let Err(error) = jetsam_core::cpu::ensure_production_hardware() {
        eprintln!("fatal: {error}");
        std::process::exit(1);
    }
    if cli.bench.is_some() {
        run_bench(&cli);
        return;
    }
    if let Err(e) = mine(&cli) {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{
        search_nonce, template_search_deadline, Block128, BlockTemplateResponse, CHUNK_SIZE,
        NONCE_REGION_SHIFT, POW_HEADER_FIELD_COUNT,
    };

    #[test]
    fn nonce_submission_is_canonical_little_endian_hex() {
        let nonce = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
        let encoded = hex::encode(nonce.to_le_bytes());
        assert_eq!(encoded.len(), 32);
        assert_eq!(
            u128::from_le_bytes(hex::decode(encoded).unwrap().try_into().unwrap()),
            nonce
        );
    }

    #[test]
    fn template_deadline_reserves_submit_margin() {
        let received_at = Instant::now();
        let deadline = template_search_deadline(received_at, 30);
        assert_eq!(
            deadline.duration_since(received_at),
            Duration::from_secs(29)
        );
        assert_eq!(template_search_deadline(received_at, 1), received_at);
        assert_eq!(template_search_deadline(received_at, 0), received_at);
    }

    #[test]
    fn expired_template_stops_before_hashing_and_keeps_its_place() {
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let mut cursor = 4_242u128;
        assert_eq!(
            search_nonce(&fields, &[0xff; 32], Instant::now(), &mut cursor),
            None
        );
        assert_eq!(
            cursor, 4_242,
            "an expired pass hashed nothing, so it must not skip that range"
        );
    }

    #[test]
    fn the_cursor_never_re_searches_a_range_it_already_covered() {
        // An all-ones target accepts the first nonce tried, so each pass ends
        // exactly at the point the search resumed from.
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let far = Instant::now() + Duration::from_secs(30);
        let mut cursor = 700u128;

        let first = search_nonce(&fields, &[0xff; 32], far, &mut cursor).expect("easy target");
        assert_eq!(first, 700);

        let second = search_nonce(&fields, &[0xff; 32], far, &mut cursor).expect("easy target");
        assert_eq!(
            second,
            700 + CHUNK_SIZE,
            "the second pass restarted inside the range the first pass had covered"
        );
    }

    #[test]
    fn a_second_miner_in_the_same_region_starts_somewhere_else() {
        // The pool derives a region from the client address, so two processes
        // on one machine get the same one. Starting both at the region's first
        // nonce would make them hash identical work forever.
        let offset = |clock: u128, pid: u128| ((clock ^ (pid << 12)) & 0xFFFF_FFFF) << 64;
        let region = u128::from(7u32) << NONCE_REGION_SHIFT;

        let first = region | offset(1_111, 4_242);
        let second = region | offset(1_111, 9_999);
        assert_ne!(first, second, "same clock, different pid must diverge");
        assert!(
            first.abs_diff(second) > u128::from(CHUNK_SIZE) * 1_000,
            "the two starts must be far more than a search apart"
        );

        // Whatever the offset, the search stays inside its own region.
        let highest = region | offset(u128::MAX, u128::MAX);
        assert_eq!(highest >> NONCE_REGION_SHIFT, 7);
        let next_region = u128::from(8u32) << NONCE_REGION_SHIFT;
        assert!(
            (next_region - highest) / CHUNK_SIZE > 1_000_000_000_000u128,
            "no worker can walk out of its region"
        );
    }

    #[test]
    fn a_pool_region_is_wider_than_any_machine_can_search() {
        // Neighbouring regions are 2^96 apart and a pass advances by
        // CHUNK_SIZE, so no worker can ever walk into its neighbour's range.
        let first = u128::from(0u32) << NONCE_REGION_SHIFT;
        let second = u128::from(1u32) << NONCE_REGION_SHIFT;
        assert_eq!(second - first, 1u128 << 96);
        assert!((second - first) / CHUNK_SIZE > 1_000_000_000_000_000_000_000u128);
        // The last region must still fit: u32::MAX << 96 is the final start.
        assert_eq!(
            u128::from(u32::MAX) << NONCE_REGION_SHIFT,
            u128::MAX - ((1u128 << 96) - 1)
        );
    }

    #[test]
    fn a_template_without_a_pool_region_still_parses() {
        let solo = r#"{"template_id":"ab","pow_fields_hex":"00","nonce_field_index":0,
            "difficulty_target_hex":"ff","height":7,"expires_in_seconds":120,"n_txs":0}"#;
        let parsed: BlockTemplateResponse = serde_json::from_str(solo).expect("node template");
        assert_eq!(parsed.nonce_prefix, None);

        let pooled = r#"{"template_id":"ab","pow_fields_hex":"00","nonce_field_index":0,
            "difficulty_target_hex":"ff","height":7,"expires_in_seconds":120,"n_txs":0,
            "nonce_prefix":3}"#;
        let parsed: BlockTemplateResponse = serde_json::from_str(pooled).expect("pool template");
        assert_eq!(parsed.nonce_prefix, Some(3));
    }
}
