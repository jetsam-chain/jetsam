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

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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

    /// Logical CPU threads used by PoW, as `jetsam --cpu-threads`: from 1 to
    /// the CPUs visible to the process. Default: every one of them.
    #[arg(long, value_name = "N", conflicts_with = "threads")]
    cpu_threads: Option<usize>,

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

    /// Benchmark this machine's walked search rate for N seconds and exit,
    /// without touching the network. Reports hashes/second at the chosen
    /// --cpu-threads, so you can size how many threads each CPU needs to pull
    /// its weight.
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
    /// `true` when a valid nonce must also satisfy the cache-resident walk:
    /// `TowerWalk(Poseidon2b(POWHDR__, patched)) < difficulty_target`.
    ///
    /// The NODE decides this, never the miner. The activation height is a
    /// consensus constant and hardcoding it here would mean mining an invalid
    /// chain the day it moved. Absent means `false`, so this binary keeps working
    /// against a node that has never heard of the field.
    #[serde(default)]
    pow_walk: bool,
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
        // A pool cannot see how fast a worker hashes: it only observes template
        // requests, and a solution is far too rare to measure anything from.
        // Reporting the rate the worker actually measured is the only honest
        // source. A node ignores the header, so nothing else changes.
        let measured = MEASURED_HASHES_PER_SECOND.load(Ordering::Relaxed);
        if measured > 0 {
            req = req.header("X-Jetsam-Hashrate", measured.to_string());
        }
        // The binary declares its own version — what is actually running, not
        // what a file on disk says — so the pool can show which release each
        // machine is on. A node ignores the header.
        req = req.header("X-Jetsam-Version", env!("CARGO_PKG_VERSION"));
        // What this binary can search under. A pool serving a walked template
        // tells walkers from pre-fork miners by this header alone, and a miner
        // that omits it is listed with the GPUs that will never find a block —
        // measured on the test pool on 2026-09-28: this binary walked at
        // 280 H/s on one thread and was still shown as `pow = None`. A node
        // ignores the header.
        req = req.header("X-Jetsam-PoW", "walk");
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

/// Hashes per second this worker last measured over a full search pass.
/// Reported to the pool, which has no other way to know what a machine weighs.
static MEASURED_HASHES_PER_SECOND: AtomicU64 = AtomicU64::new(0);

const CHUNK_SIZE: u128 = 10_000_000;
const DIGEST_BATCH: usize = 256;
/// Batch size once the node asks for the cache-resident walk. See `search_nonce`.
const WALK_BATCH: usize = 4;
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
/// Record what this pass actually achieved, so the next request can tell the
/// pool what this machine weighs. Passes shorter than a tenth of a second say
/// nothing useful and are ignored.
fn publish_rate(hashed: &AtomicU64, started: Instant) {
    let elapsed = started.elapsed().as_secs_f64();
    let done = hashed.load(Ordering::Relaxed);
    if elapsed >= 0.1 && done > 0 {
        MEASURED_HASHES_PER_SECOND.store((done as f64 / elapsed) as u64, Ordering::Relaxed);
    }
}

/// `cursor` is where this worker's search resumes, and it only ever moves
/// forward. A template that expires mid-search is fetched again and searched
/// again; restarting from a fresh point re-hashed the range the previous pass
/// had already covered. Raising `new_work` ends the search like the deadline.
fn search_nonce(
    pow_fields: &[Block128; POW_HEADER_FIELD_COUNT],
    target: &[u8; 32],
    deadline: Instant,
    cursor: &mut u128,
    pow_walk: bool,
    new_work: &AtomicBool,
) -> Option<u128> {
    let num_threads = rayon::current_num_threads();
    let per_thread = CHUNK_SIZE.div_ceil(num_threads as u128);

    let mut chunk_start = *cursor;
    // Counted, not derived from the cursor: a pass cut short by the deadline
    // leaves its last chunk unfinished, and charging the worker for the whole
    // chunk would report a machine as faster than it is.
    let hashed = AtomicU64::new(0);
    let started = Instant::now();
    // Raised by the thread that finds a solution, so the others stop within a
    // batch. Each thread otherwise runs its share of the chunk to the end, or
    // to the deadline, before the solution is handed back — and on a machine
    // that cannot finish a chunk inside the search budget that is the deadline
    // every time, where the mining loop discards the solution as too late.
    let found = AtomicBool::new(false);

    loop {
        if Instant::now() >= deadline || new_work.load(Ordering::Relaxed) {
            *cursor = chunk_start;
            publish_rate(&hashed, started);
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
            // One 512 KiB scratchpad per thread, allocated once for the whole
            // chunk. Allocating one per nonce would cost more than the walk.
            let mut scratch = pow_walk.then(jetsam_poseidon2b::towerwalk::Scratch::new);
            // A batch is the deadline-check granularity. 256 sponge digests is a
            // few milliseconds; 256 walked digests would be half a second, long
            // enough to keep grinding a template that has already expired.
            let batch = if pow_walk { WALK_BATCH } else { DIGEST_BATCH };
            let mut nonce = ts;
            while nonce < te {
                if found.load(Ordering::Relaxed)
                    || new_work.load(Ordering::Relaxed)
                    || Instant::now() >= deadline
                {
                    return None;
                }
                let n = ((te - nonce).min(batch as u128)) as usize;
                hasher.hash_into(nonce, &mut digests[..n]);
                hashed.fetch_add(n as u64, Ordering::Relaxed);
                for (i, seed) in digests[..n].iter().enumerate() {
                    let hash = match scratch.as_mut() {
                        Some(sc) => {
                            jetsam_poseidon2b::towerwalk::towerwalk_digest_with(sc, seed)
                        }
                        None => *seed,
                    };
                    if le256_lt(&hash, target) {
                        found.store(true, Ordering::Relaxed);
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
            publish_rate(&hashed, started);
            return solution;
        }
    }
}

/// How often a search asks the node whether its work changed.
const WORK_CHECK_INTERVAL: Duration = Duration::from_secs(2);

/// `fresh` is other work than `current`: a new tip, whose parent is part of
/// the PoW input, or work the node renewed on the same tip.
fn is_new_work(current: &BlockTemplateResponse, fresh: &BlockTemplateResponse) -> bool {
    fresh.pow_fields_hex != current.pow_fields_hex
        || fresh.difficulty_target_hex != current.difficulty_target_hex
}

/// Run `search`, asking the node every `check_every` for a template; the
/// search is stopped as soon as one carries new work, which is returned.
fn search_until_new_work(
    rpc: &RpcClient,
    coinbase: &str,
    current: &BlockTemplateResponse,
    check_every: Duration,
    search: impl FnOnce(&AtomicBool) -> Option<u128> + Send,
) -> (Option<u128>, Option<(BlockTemplateResponse, Instant)>) {
    let new_work = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| search(&new_work));
        let mut fresh = None;
        let mut next_check = Instant::now() + check_every;
        while !worker.is_finished() {
            if Instant::now() < next_check {
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }
            if let Ok(template) = rpc.get_template(coinbase) {
                if is_new_work(current, &template) {
                    fresh = Some((template, Instant::now()));
                    new_work.store(true, Ordering::Relaxed);
                    break;
                }
            }
            next_check = Instant::now() + check_every;
        }
        (worker.join().expect("PoW search thread"), fresh)
    })
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

/// The PoW pool width, or `None` for every logical CPU the process can see.
fn pow_threads(cli: &Cli, available: usize) -> Result<Option<usize>> {
    match cli.cpu_threads {
        Some(requested) if requested == 0 || requested > available => Err(anyhow!(
            "--cpu-threads {requested} is outside the host-visible range 1..={available}"
        )),
        Some(requested) => Ok(Some(requested)),
        None => Ok((cli.threads > 0).then_some(cli.threads)),
    }
}

fn configure_pow_pool(cli: &Cli) -> Result<()> {
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    if let Some(threads) = pow_threads(cli, available)? {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("configure PoW thread pool")?;
    }
    Ok(())
}

fn mine(cli: &Cli) -> Result<()> {
    let rpc = RpcClient::new(&cli.rpc, cli.key.clone());

    configure_pow_pool(cli)?;
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

    let mut next: Option<(BlockTemplateResponse, Instant)> = None;
    loop {
        // Fetch template, unless the search already brought back the new tip's.
        let (tmpl, template_received_at) = match next.take() {
            Some(fresh) => fresh,
            None => match rpc.get_template(&cli.coinbase) {
                Ok(t) => (t, Instant::now()),
                Err(e) => {
                    eprintln!("template fetch failed: {e}  — retrying in 2s");
                    std::thread::sleep(Duration::from_secs(2));
                    continue;
                }
            },
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

        let (found, fresh) = search_until_new_work(
            &rpc,
            &cli.coinbase,
            &tmpl,
            WORK_CHECK_INTERVAL,
            |new_work| {
                search_nonce(
                    &pow_fields,
                    &target,
                    deadline,
                    &mut cursor,
                    tmpl.pow_walk,
                    new_work,
                )
            },
        );
        // The node may hold a single template per worker: the one the check
        // fetched is the next one mined, never fetched twice.
        next = fresh;
        let nonce = match found {
            Some(n) => n,
            None => {
                match &next {
                    Some((new, _)) => {
                        eprintln!("└─ NEW WORK  h={}  switching template", new.height)
                    }
                    None => eprintln!("└─ EXPIRED  refreshing node-owned template"),
                }
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

/// Hashes per second, on the current pool, of the work `search_nonce` does for
/// each nonce of a walked template: the sponge digest, then the walk.
fn bench_rate(duration: Duration) -> f64 {
    let threads = rayon::current_num_threads();
    // A fixed dummy header; the hash rate does not depend on its contents.
    let fields = [Block128::from(0x9e37_79b9_7f4a_7c15u128); POW_HEADER_FIELD_COUNT];
    let started = Instant::now();
    let deadline = started + duration;
    let total: u64 = (0..threads)
        .into_par_iter()
        .map(|tid| {
            let mut hasher = FixedFieldNonceBatch::new(TAG_POWHDR, &fields, POW_NONCE_FIELD_INDEX);
            let mut digests = [[0u8; 32]; WALK_BATCH];
            let mut scratch = jetsam_poseidon2b::towerwalk::Scratch::new();
            let mut nonce = (tid as u128) << 96;
            let mut done: u64 = 0;
            while Instant::now() < deadline {
                hasher.hash_into(nonce, &mut digests);
                for seed in &digests {
                    std::hint::black_box(jetsam_poseidon2b::towerwalk::towerwalk_digest_with(
                        &mut scratch,
                        seed,
                    ));
                }
                nonce = nonce.wrapping_add(WALK_BATCH as u128);
                done += WALK_BATCH as u64;
            }
            done
        })
        .sum();
    total as f64 / started.elapsed().as_secs_f64()
}

fn run_bench(cli: &Cli) -> Result<()> {
    configure_pow_pool(cli)?;
    let threads = rayon::current_num_threads();
    let secs = cli.bench.unwrap_or(10).max(1);
    eprintln!("benchmarking the walked search on {threads} thread(s) for {secs}s...");
    let hps = bench_rate(Duration::from_secs(secs));
    println!(
        "walked search: {:.2} kH/s  ({threads} threads, {:.3} kH/s per thread)",
        hps / 1.0e3,
        hps / 1.0e3 / threads as f64
    );
    println!(
        "Each nonce is a sponge digest and its walk, as every current network asks. \
         This is an upper bound, measured with nothing else running: a node building \
         and proving blocks on the same cores lowers the rate reached while mining."
    );
    Ok(())
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
        if let Err(e) = run_bench(&cli) {
            eprintln!("fatal: {e}");
            std::process::exit(1);
        }
        return;
    }
    if let Err(e) = mine(&cli) {
        eprintln!("fatal: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use std::io::{BufRead, Read, Write};
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    use super::{
        bench_rate, pow_threads, search_nonce, search_until_new_work, template_search_deadline,
        Block128, BlockTemplateResponse, Cli, RpcClient, CHUNK_SIZE, NONCE_REGION_SHIFT,
        POW_HEADER_FIELD_COUNT,
    };
    use clap::Parser;

    /// A search never told of new work.
    static NEVER: AtomicBool = AtomicBool::new(false);

    const TEMPLATE_AT_7: &str = r#"{"template_id":"a7","pow_fields_hex":"07","nonce_field_index":0,
        "difficulty_target_hex":"00","height":7,"expires_in_seconds":240,"n_txs":1,
        "template_seq":1,"parent_id":"66","ttl_remaining_ms":240000}"#;
    /// The same work served again, later.
    const TEMPLATE_AT_7_AGAIN: &str = r#"{"template_id":"a7","pow_fields_hex":"07",
        "nonce_field_index":0,"difficulty_target_hex":"00","height":7,"expires_in_seconds":240,
        "n_txs":1,"template_seq":1,"parent_id":"66","ttl_remaining_ms":180000}"#;
    /// New work on the same parent, once the previous work expired at the node.
    /// A node's template ids repeat, so the id says nothing.
    const TEMPLATE_AT_7_RENEWED: &str = r#"{"template_id":"a7","pow_fields_hex":"17",
        "nonce_field_index":0,"difficulty_target_hex":"00","height":7,"expires_in_seconds":240,
        "n_txs":1,"template_seq":2,"parent_id":"66","ttl_remaining_ms":240000}"#;
    const TEMPLATE_AT_8: &str = r#"{"template_id":"a8","pow_fields_hex":"08","nonce_field_index":0,
        "difficulty_target_hex":"00","height":8,"expires_in_seconds":240,"n_txs":1,
        "template_seq":3,"parent_id":"77","ttl_remaining_ms":240000}"#;

    /// One search thread: the stop is checked once per batch, and a debug
    /// build on a loaded machine makes a batch on every core slow.
    fn one_thread() -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool")
    }

    /// A JSON-RPC node that answers `first` to the first request and `then`
    /// to every later one.
    fn fake_node(first: &'static str, then: &'static str) -> RpcClient {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for (served, stream) in listener.incoming().enumerate() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse().unwrap();
                    }
                }
                let mut body = vec![0; length];
                let _ = reader.read_exact(&mut body);
                let result = if served == 0 { first } else { then };
                let reply = format!(r#"{{"jsonrpc":"2.0","id":1,"result":{result}}}"#);
                let _ = write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\
                     content-length: {}\r\nconnection: close\r\n\r\n{reply}",
                    reply.len()
                );
            }
        });
        RpcClient::new(&url, None)
    }

    /// The miner kept searching a template whose parent had been replaced,
    /// until the template expired: up to 240 s of work no block could use.
    #[test]
    fn a_search_switches_to_the_new_tip_instead_of_running_to_expiry() {
        let rpc = fake_node(TEMPLATE_AT_7, TEMPLATE_AT_8);
        let current = rpc.get_template("").unwrap();
        assert_eq!(current.height, 7);
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let mut cursor = 0u128;
        let deadline = Instant::now() + Duration::from_secs(8);
        let started = Instant::now();

        let (found, fresh) =
            search_until_new_work(&rpc, "", &current, Duration::from_millis(100), |moved| {
                one_thread().install(|| {
                    search_nonce(&fields, &[0u8; 32], deadline, &mut cursor, false, moved)
                })
            });

        assert_eq!(found, None);
        assert!(
            started.elapsed() < Duration::from_secs(4),
            "the search ran {:?} on a replaced parent",
            started.elapsed()
        );
        assert_eq!(fresh.expect("the new tip's template").0.height, 8);
    }

    /// Walked digests per second on the current pool's first thread.
    fn walk_rate() -> f64 {
        let mut scratch = jetsam_poseidon2b::towerwalk::Scratch::new();
        let mut seed = [7u8; 32];
        let started = Instant::now();
        let mut walked = 0u32;
        while walked < 3 || started.elapsed() < Duration::from_millis(500) {
            seed = jetsam_poseidon2b::towerwalk::towerwalk_digest_with(&mut scratch, &seed);
            walked += 1;
        }
        f64::from(walked) / started.elapsed().as_secs_f64()
    }

    /// `--bench` timed the sponge digest alone and printed it as the miner's
    /// rate, while every current network has the miner walk each digest: the
    /// figure was some 80 times what a walking miner searches.
    #[test]
    fn the_bench_times_the_walked_search_the_miner_runs() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool");
        let before = pool.install(walk_rate);
        let bench = pool.install(|| bench_rate(Duration::from_millis(1_500)));
        let after = pool.install(walk_rate);

        assert!(
            bench <= 3.0 * before.max(after),
            "bench {bench:.0} H/s against a walk of {before:.0}-{after:.0} H/s"
        );
        assert!(
            bench >= before.min(after) / 3.0,
            "bench {bench:.0} H/s against a walk of {before:.0}-{after:.0} H/s"
        );
    }

    fn threads_for(args: &[&str], available: usize) -> anyhow::Result<Option<usize>> {
        let cli = Cli::try_parse_from(std::iter::once("jetsam-miner").chain(args.iter().copied()))?;
        pow_threads(&cli, available)
    }

    /// `--cpu-threads` reads as in `jetsam`: the logical CPUs given to PoW,
    /// from one to every CPU this process can see.
    #[test]
    fn cpu_threads_sizes_the_search_as_the_node_reads_it() {
        assert_eq!(threads_for(&["--cpu-threads", "4"], 8).unwrap(), Some(4));
        assert_eq!(threads_for(&["--cpu-threads", "8"], 8).unwrap(), Some(8));
        for outside in ["0", "9"] {
            let error = threads_for(&["--cpu-threads", outside], 8).unwrap_err();
            assert!(error.to_string().contains("1..=8"), "{error}");
        }
        assert!(threads_for(&["--cpu-threads", "4", "--threads", "2"], 8).is_err());
        assert_eq!(threads_for(&["--threads", "3"], 8).unwrap(), Some(3));
        assert_eq!(threads_for(&[], 8).unwrap(), None);
    }

    /// With its own coinbase a worker holds the node's only template: work the
    /// node renews on the same tip, fetched by the check, has to be the next
    /// one mined, or the node refuses every other request until it expires.
    #[test]
    fn new_work_on_the_same_tip_is_taken_too() {
        let rpc = fake_node(TEMPLATE_AT_7, TEMPLATE_AT_7_RENEWED);
        let current = rpc.get_template("").unwrap();
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let mut cursor = 0u128;
        let deadline = Instant::now() + Duration::from_secs(8);
        let started = Instant::now();

        let (found, fresh) =
            search_until_new_work(&rpc, "", &current, Duration::from_millis(100), |moved| {
                one_thread().install(|| {
                    search_nonce(&fields, &[0u8; 32], deadline, &mut cursor, false, moved)
                })
            });

        assert_eq!(found, None);
        assert!(started.elapsed() < Duration::from_secs(4));
        assert_eq!(fresh.expect("the renewed work").0.pow_fields_hex, "17");
    }

    /// The same work served again does not interrupt the search.
    #[test]
    fn the_same_work_served_again_does_not_interrupt_the_search() {
        let rpc = fake_node(TEMPLATE_AT_7, TEMPLATE_AT_7_AGAIN);
        let current = rpc.get_template("").unwrap();
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let mut cursor = 0u128;
        let deadline = Instant::now() + Duration::from_millis(1_500);

        let (found, fresh) =
            search_until_new_work(&rpc, "", &current, Duration::from_millis(100), |moved| {
                one_thread().install(|| {
                    search_nonce(&fields, &[0u8; 32], deadline, &mut cursor, false, moved)
                })
            });

        assert_eq!(found, None);
        assert!(fresh.is_none());
        assert!(Instant::now() >= deadline);
    }

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
            search_nonce(
                &fields,
                &[0xff; 32],
                Instant::now(),
                &mut cursor,
                false,
                &NEVER
            ),
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
        //
        // The chunk is cut into one slice per rayon thread and collected with
        // `find_map_any`, which returns *a* solution, not the lowest one.
        // Against a target every nonce meets, every slice starts on a solution
        // at once, so on the shared pool the nonce that came back was whichever
        // thread won the race — 256 slices on this machine, 256 possible
        // answers, and the assertion below held only when slice zero won.
        // Pinning a one-thread pool leaves a single slice, which is what makes
        // these two nonces the only ones the search can return.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool");
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let far = Instant::now() + Duration::from_secs(30);
        let mut cursor = 700u128;

        let first = pool
            .install(|| search_nonce(&fields, &[0xff; 32], far, &mut cursor, false, &NEVER))
            .expect("easy target");
        assert_eq!(first, 700);

        let second = pool
            .install(|| search_nonce(&fields, &[0xff; 32], far, &mut cursor, false, &NEVER))
            .expect("easy target");
        assert_eq!(
            second,
            700 + CHUNK_SIZE,
            "the second pass restarted inside the range the first pass had covered"
        );
    }

    #[test]
    fn a_pass_cut_short_by_the_deadline_still_moves_the_cursor_forward() {
        // An all-zero target is never met, so this pass can only end on the
        // deadline — long before a 10M-nonce chunk is finished on any machine.
        // Resuming from the same place would hash the same nonces again on
        // every pass; the cursor must move past the chunk that was started.
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let start = 5_000u128;
        let mut cursor = start;
        // Spawning the thread pool takes longer than a short deadline; in
        // production it is long warm by the time a template arrives.
        rayon::current_num_threads();
        let soon = Instant::now() + Duration::from_millis(300);
        assert_eq!(
            search_nonce(&fields, &[0u8; 32], soon, &mut cursor, false, &NEVER),
            None
        );
        assert!(
            cursor > start,
            "the cursor went back to {cursor} after a pass that started at {start}"
        );
        assert_eq!(cursor, start + CHUNK_SIZE, "the interrupted chunk must be skipped whole");
    }

    #[test]
    fn a_solution_surfaces_as_soon_as_one_thread_finds_it() {
        // Two threads split a chunk into halves of 5M nonces. The first half
        // starts one nonce before a known solution, so its thread finds it
        // within the first batch. One nonce in 2^28 meets this target, so the
        // other half holds nothing its thread could reach in seconds: left
        // alone it hashes on until the deadline. The solution must not wait
        // for it. When a machine cannot finish a chunk inside the search
        // budget, waiting hands every solution back exactly at the deadline,
        // where the mining loop discards it as too late to submit.
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let mut target = [0u8; 32];
        target[28] = 0x10; // hash < 2^228
        const KNOWN_SOLUTION: u128 = 304_511_652;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("two-thread pool");
        let mut cursor = KNOWN_SOLUTION - 1;
        let deadline = Instant::now() + Duration::from_secs(3);
        let found =
            pool.install(|| search_nonce(&fields, &target, deadline, &mut cursor, false, &NEVER));
        let returned_at = Instant::now();

        assert_eq!(found, Some(KNOWN_SOLUTION));
        assert!(
            returned_at < deadline,
            "the solution was found within milliseconds but only surfaced at the deadline"
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

    /// A node that has never heard of the walk must keep this binary working,
    /// and a node that asks for it must be obeyed. Getting the default wrong in
    /// either direction means mining a chain nobody else accepts.
    #[test]
    fn the_walk_is_off_unless_the_node_asks_for_it() {
        let without = r#"{"template_id":"ab","pow_fields_hex":"00","nonce_field_index":0,
            "difficulty_target_hex":"ff","height":7,"expires_in_seconds":120,"n_txs":0}"#;
        let parsed: BlockTemplateResponse = serde_json::from_str(without).expect("old node");
        assert!(!parsed.pow_walk, "absent must mean off");

        let with = r#"{"template_id":"ab","pow_fields_hex":"00","nonce_field_index":0,
            "difficulty_target_hex":"ff","height":7,"expires_in_seconds":120,"n_txs":0,
            "pow_walk":true}"#;
        let parsed: BlockTemplateResponse = serde_json::from_str(with).expect("forked node");
        assert!(parsed.pow_walk);
    }

    /// With the walk on, a trivial target is still satisfied — the search path
    /// reaches the walk and compares its output, not the seed.
    ///
    /// A target of all-ones is met by any digest, so this cannot pass by
    /// accident of which value is compared; what it proves is that the walked
    /// path runs at all and returns a nonce. One walked hash is ~1.9 ms, so this
    /// stays a unit test rather than a benchmark.
    #[test]
    fn the_walked_search_finds_a_nonce_under_a_trivial_target() {
        let fields = [Block128::from(0u128); POW_HEADER_FIELD_COUNT];
        let mut cursor = 0u128;
        let far = Instant::now() + Duration::from_secs(60);
        let nonce = search_nonce(&fields, &[0xff; 32], far, &mut cursor, true, &NEVER)
            .expect("any digest beats an all-ones target");
        // And the walk really is applied, not skipped: the digest it produces
        // for that nonce differs from the seed it was computed from.
        let mut hasher = super::FixedFieldNonceBatch::new(
            super::TAG_POWHDR,
            &fields,
            super::POW_NONCE_FIELD_INDEX,
        );
        let mut seeds = [[0u8; 32]; 1];
        hasher.hash_into(nonce, &mut seeds);
        let walked = jetsam_poseidon2b::towerwalk::towerwalk_digest(&seeds[0]);
        assert_ne!(walked, seeds[0], "the walk must not be the identity");
    }
}
