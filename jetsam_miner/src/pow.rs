// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.
// Portions derived from an Apache-2.0 licensed upstream; see NOTICE.

//! Parallel Poseidon2b PoW nonce search.
//!
//! PoW is computed over the fixed semantic header field schedule. Detached
//! nonce-independent witness data is prepared before this phase. The search
//! then owns the miner's shared all-core pool until it finds a nonce or is
//! cancelled. Latency-sensitive wallet proof and admission verification also
//! ask the search to drain at its next batch boundary; HistoryStep proving
//! starts only after the search has drained.
//!

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use jetsam_chain::block_header::BlockHeader;
use jetsam_chain::consensus::difficulty::le256_lt;
use jetsam_chain::consensus::pow::{pow_header_fields, PowNonceBatchHasher};

/// Result of a successful PoW search.
#[derive(Debug, Clone)]
pub struct PowSolution {
    pub nonce: u128,
    pub pow_hash: [u8; 32],
}

// ---------------------------------------------------------------------------
// Hashrate meter
// ---------------------------------------------------------------------------

/// How often the built-in miner reports its measured rate while searching.
const REPORT_INTERVAL: Duration = Duration::from_secs(15);

/// How long a published rate remains a statement about the present.
///
/// A mining node alternates between proving — where no hash is computed at all
/// — and nonce search. Once the search stops, the last window describes the
/// past; reporting it as the current rate would be a lie the operator cannot
/// see through.
const RATE_MAX_AGE_MS: u64 = 120_000;

/// Process-wide PoW counters for the built-in miner.
///
/// The process runs one search at a time, so a single global meter describes it
/// exactly. Readers never block the search: every field is a relaxed atomic.
struct PowMeter {
    total_hashes: AtomicU64,
    /// Search time folded into the open window, in nanoseconds.
    window_nanos: AtomicU64,
    /// Hashes folded into the open window.
    window_hashes: AtomicU64,
    /// Last completed window's rate in hashes per second, as `f64` bits.
    rate_bits: AtomicU64,
    /// Unix milliseconds at which `rate_bits` was published. Zero = never.
    rate_at_ms: AtomicU64,
}

impl PowMeter {
    const fn new() -> Self {
        Self {
            total_hashes: AtomicU64::new(0),
            window_nanos: AtomicU64::new(0),
            window_hashes: AtomicU64::new(0),
            rate_bits: AtomicU64::new(0),
            rate_at_ms: AtomicU64::new(0),
        }
    }

    fn add_hashes(&self, n: u64) {
        self.total_hashes.fetch_add(n, Ordering::Relaxed);
    }

    fn total_hashes(&self) -> u64 {
        self.total_hashes.load(Ordering::Relaxed)
    }

    /// Fold one slice of search time into the open window.
    ///
    /// The window deliberately spans block boundaries: a chain whose blocks are
    /// found in under a second would otherwise never accumulate enough search
    /// time to report anything. Returns the rate when the window closes.
    fn accumulate(&self, hashes: u64, elapsed: Duration) -> Option<f64> {
        let nanos = self
            .window_nanos
            .fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed)
            .saturating_add(elapsed.as_nanos() as u64);
        let total = self
            .window_hashes
            .fetch_add(hashes, Ordering::Relaxed)
            .saturating_add(hashes);
        if nanos < REPORT_INTERVAL.as_nanos() as u64 {
            return None;
        }
        self.window_nanos.store(0, Ordering::Relaxed);
        self.window_hashes.store(0, Ordering::Relaxed);
        let rate = window_rate(total, Duration::from_nanos(nanos))?;
        self.publish(rate, now_unix_ms());
        Some(rate)
    }

    fn publish(&self, hps: f64, at_unix_ms: u64) {
        self.rate_bits.store(hps.to_bits(), Ordering::Relaxed);
        self.rate_at_ms.store(at_unix_ms, Ordering::Relaxed);
    }

    fn rate_hps(&self, now_unix_ms: u64) -> Option<f64> {
        let at = self.rate_at_ms.load(Ordering::Relaxed);
        if at == 0 || now_unix_ms.saturating_sub(at) >= RATE_MAX_AGE_MS {
            return None;
        }
        let hps = f64::from_bits(self.rate_bits.load(Ordering::Relaxed));
        (hps.is_finite() && hps > 0.0).then_some(hps)
    }
}

static POW_METER: PowMeter = PowMeter::new();

/// Serialises the tests that observe the process-wide [`POW_METER`].
///
/// One of them asserts the counter is unchanged across a call; another runs a
/// real search, which moves it. Cargo runs both on threads of one process, so
/// without this they race and the assertion fails for a reason that has nothing
/// to do with what it tests.
#[cfg(test)]
static METER_OBSERVERS: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Measured PoW rate of this process's built-in miner, in hashes per second.
///
/// `None` when the node is not mining, or when it is currently proving a block
/// rather than searching for a nonce.
pub fn local_hashrate_hps() -> Option<f64> {
    POW_METER.rate_hps(now_unix_ms())
}

/// Total PoW hashes computed by this process's built-in miner since start.
pub fn local_pow_hashes() -> u64 {
    POW_METER.total_hashes()
}

/// Render a hashrate the way a miner reads it: three significant digits and the
/// unit that keeps the number between 1 and 1000.
pub fn format_hashrate(hps: f64) -> String {
    const UNITS: [&str; 6] = ["H/s", "kH/s", "MH/s", "GH/s", "TH/s", "PH/s"];
    if !hps.is_finite() || hps <= 0.0 {
        return "0 H/s".to_string();
    }
    let mut scaled = hps;
    let mut unit = 0;
    while scaled >= 1000.0 && unit + 1 < UNITS.len() {
        scaled /= 1000.0;
        unit += 1;
    }
    let decimals = if scaled < 10.0 {
        2
    } else if scaled < 100.0 {
        1
    } else {
        0
    };
    format!("{scaled:.decimals$} {}", UNITS[unit])
}

/// Rate over one measurement window, or `None` when the window says nothing.
fn window_rate(hashes: u64, elapsed: Duration) -> Option<f64> {
    let seconds = elapsed.as_secs_f64();
    if hashes == 0 || !(seconds > 0.0) {
        return None;
    }
    Some(hashes as f64 / seconds)
}

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Stand-alone benchmark
// ---------------------------------------------------------------------------

/// What one TowerHash benchmark measured.
#[derive(Debug, Clone, Copy)]
pub struct BenchResult {
    /// Threads that actually ran the search.
    pub threads: usize,
    /// Wall-clock seconds the benchmark ran.
    pub seconds: f64,
    /// Hashes computed in that time.
    pub hashes: u64,
}

impl BenchResult {
    /// Whole-machine rate in hashes per second.
    pub fn hashes_per_second(&self) -> f64 {
        window_rate(self.hashes, Duration::from_secs_f64(self.seconds)).unwrap_or(0.0)
    }

    /// Rate of a single thread — the number that says how many threads a given
    /// CPU needs to pull its weight.
    pub fn per_thread_hps(&self) -> f64 {
        if self.threads == 0 {
            return 0.0;
        }
        self.hashes_per_second() / self.threads as f64
    }
}

/// Measure this machine's TowerHash rate, touching no chain data, no wallet and
/// no network.
///
/// `threads` selects the width; `None` uses every logical CPU the process can
/// see. The benchmark runs in its own Rayon pool and deliberately does not feed
/// the mining counters: a benchmark is not mining.
pub fn bench_towerhash(duration: Duration, threads: Option<usize>) -> BenchResult {
    use jetsam_chain::consensus::pow::POW_HEADER_FIELD_COUNT;
    use rayon::prelude::*;

    const DIGEST_BATCH: usize = 256;

    let width = threads
        .filter(|n| *n > 0)
        .unwrap_or_else(rayon::current_num_threads)
        .max(1);

    // A fixed arbitrary header. TowerHash cost is independent of its contents.
    let fields = [jetsam_core::Block128::from(0x9e37_79b9_7f4a_7c15u128); POW_HEADER_FIELD_COUNT];

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(width)
        .build()
        .expect("benchmark thread pool");

    let started = Instant::now();
    let deadline = started + duration;
    let hashes: u64 = pool.install(|| {
        (0..width)
            .into_par_iter()
            .map(|thread_id| {
                let mut hasher = PowNonceBatchHasher::new(&fields);
                let mut digests = [[0u8; 32]; DIGEST_BATCH];
                // Separate the threads' nonce ranges so no two repeat work.
                let mut nonce = (thread_id as u128) << 96;
                let mut done: u64 = 0;
                while Instant::now() < deadline {
                    hasher.hash_into(nonce, &mut digests);
                    nonce = nonce.wrapping_add(DIGEST_BATCH as u128);
                    done += DIGEST_BATCH as u64;
                }
                done
            })
            .sum()
    });

    BenchResult {
        threads: width,
        seconds: started.elapsed().as_secs_f64(),
        hashes,
    }
}

/// Tracks what this search has contributed since the last fold.
///
/// Only the time between ticks counts, so the proving phase between two
/// searches never dilutes the measured rate.
struct SearchTicker {
    since: Instant,
    base_hashes: u64,
}

impl SearchTicker {
    fn start() -> Self {
        Self {
            since: Instant::now(),
            base_hashes: POW_METER.total_hashes(),
        }
    }

    /// Fold everything since the last tick into the global window.
    fn tick(&mut self) -> Option<f64> {
        let elapsed = self.since.elapsed();
        let hashes = POW_METER.total_hashes().saturating_sub(self.base_hashes);
        *self = Self::start();
        POW_METER.accumulate(hashes, elapsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn formats_a_rate_in_the_unit_an_operator_reads() {
        assert_eq!(format_hashrate(0.0), "0 H/s");
        assert_eq!(format_hashrate(412.4), "412 H/s");
        assert_eq!(format_hashrate(15_300.0), "15.3 kH/s");
        assert_eq!(format_hashrate(816_000.0), "816 kH/s");
        assert_eq!(format_hashrate(2_790_000.0), "2.79 MH/s");
        assert_eq!(format_hashrate(1_400_000_000.0), "1.40 GH/s");
    }

    #[test]
    fn a_window_rate_needs_both_time_and_work() {
        assert_eq!(
            window_rate(1_000_000, Duration::from_secs(2)),
            Some(500_000.0)
        );
        // A window with no elapsed time would divide by zero.
        assert_eq!(window_rate(1_000_000, Duration::ZERO), None);
        // A window with no work is not a measurement of zero hashrate.
        assert_eq!(window_rate(0, Duration::from_secs(2)), None);
    }

    #[test]
    fn a_published_rate_expires_when_the_search_stops() {
        let meter = PowMeter::new();
        assert_eq!(meter.rate_hps(0), None);
        meter.publish(2_790_000.0, 1_000);
        assert_eq!(meter.rate_hps(1_000), Some(2_790_000.0));
        assert_eq!(
            meter.rate_hps(1_000 + RATE_MAX_AGE_MS - 1),
            Some(2_790_000.0)
        );
        // Past the freshness horizon the node is proving, not hashing: a stale
        // number is worse than no number.
        assert_eq!(meter.rate_hps(1_000 + RATE_MAX_AGE_MS), None);
    }

    #[test]
    fn short_searches_still_add_up_to_a_published_rate() {
        // A block found in under a second is one short search. The window has to
        // survive across searches, or a fast chain would never report a rate.
        let meter = PowMeter::new();
        let slice = REPORT_INTERVAL / 10;
        for _ in 0..9 {
            assert_eq!(meter.accumulate(100_000, slice), None);
        }
        let rate = meter
            .accumulate(100_000, slice)
            .expect("ten slices of a tenth of the interval close one window");
        assert!((rate - 1_000_000.0 / REPORT_INTERVAL.as_secs_f64()).abs() < 1.0);
        // Publishing resets the window rather than double-counting its work.
        assert_eq!(meter.accumulate(100_000, slice), None);
    }

    #[test]
    fn a_bench_measures_real_work_on_the_requested_threads() {
        let result = bench_towerhash(Duration::from_millis(200), Some(2));
        assert_eq!(result.threads, 2);
        assert!(
            result.hashes > 0,
            "a bench that hashes nothing is not a bench"
        );
        assert!(result.hashes_per_second() > 0.0);
        // Per-thread rate is the whole-machine rate divided by the threads used.
        let expected = result.hashes_per_second() / 2.0;
        assert!((result.per_thread_hps() - expected).abs() < f64::EPSILON * expected.max(1.0));
    }

    #[test]
    fn a_bench_does_not_pollute_the_miner_counters() {
        let _serialised = METER_OBSERVERS.lock().unwrap_or_else(|e| e.into_inner());
        let before = local_pow_hashes();
        bench_towerhash(Duration::from_millis(50), Some(1));
        assert_eq!(local_pow_hashes(), before);
    }

    #[test]
    fn hashes_accumulate_across_windows() {
        let meter = PowMeter::new();
        assert_eq!(meter.total_hashes(), 0);
        meter.add_hashes(256);
        meter.add_hashes(256);
        assert_eq!(meter.total_hashes(), 512);
    }
}

/// Fold this slice of search time in, and tell the operator whenever a window
/// closes.
///
/// Every exit from the search reports too: on a chain whose blocks are found
/// inside a single chunk, the mid-loop call is never reached, and a miner that
/// prints nothing looks broken.
fn report_rate(ticker: &mut SearchTicker, height: u64, threads: usize) {
    if let Some(rate) = ticker.tick() {
        // What an operator checks to know the machine is working. The marker
        // and the rate come first, before anything they have to read past.
        // No colour: see `crate::colour`.
        tracing::info!(
            height,
            "{}  {}  ·  {threads} threads · {} hashes total",
            crate::colour::MINING,
            format_hashrate(rate),
            POW_METER.total_hashes(),
        );
    }
}

/// Nonces handed out on the first pass, before any measurement exists.
///
/// At module scope because a thread's share of it — `CHUNK_SIZE / threads` —
/// is what a thread must stop grinding once another has won, and the
/// regression test sizes its shares from the same constant the search uses.
const CHUNK_SIZE: u128 = 1_000_000;

/// How long one pass should take, whatever the machine.
///
/// A pass is the granularity of everything outside the search itself:
/// cancellation when the tip moves, preemption, and the rate line. A fixed
/// nonce count makes that granularity a function of how fast the miner is,
/// which is exactly backwards.
///
/// 1,000,000 nonces cost a 150-thread machine 9 seconds and a 4-thread one
/// **five and a half minutes** [MEASURED 2026-09-26 on the test chain:
/// 2.96 kH/s across four threads]. The small miner spent every pass grinding a
/// parent the network had already replaced, finished, threw the work away, and
/// started over — it could not win a block at any hashrate. Blocks come every
/// 90 seconds; a pass has to be far shorter than that or the miner is
/// permanently answering the previous question.
const CHUNK_TARGET: Duration = Duration::from_millis(500);

/// Floor and ceiling for the adaptive size.
///
/// The floor keeps per-pass overhead — building the pool's work, the atomics,
/// the rate bookkeeping — from dominating on a very slow machine. The ceiling
/// keeps a very fast one from drifting back into the behaviour this replaces.
const CHUNK_MIN: u128 = 1_024;
const CHUNK_MAX: u128 = 4_000_000;

/// Where the first pass starts, before anything has been measured.
///
/// It starts at the floor and grows, rather than starting large and shrinking.
/// Growing costs a handful of very short passes — from the floor to the
/// ceiling is six of them — while starting large costs a slow machine its
/// first **five and a half minutes** on a parent the network may already have
/// replaced. That first pass is the one that decides whether a new miner ever
/// gets into the race, so it is the one that must not be long.
const CHUNK_START: u128 = CHUNK_MIN;

/// Size the next pass from how long the last one took.
///
/// Damped on purpose: at most a quadrupling or a quartering per pass, so one
/// slow pass — a scheduler hiccup, another job landing on the machine — can
/// neither collapse the size nor blow it up. Convergence still takes a handful
/// of passes from either end.
fn next_chunk_size(current: u128, elapsed: Duration) -> u128 {
    let target = CHUNK_TARGET.as_nanos().max(1);
    let took = elapsed.as_nanos().max(1);
    let scaled = if took >= target {
        let down = (took / target).clamp(1, 4) as u128;
        current / down
    } else {
        let up = (target / took).clamp(1, 4) as u128;
        current.saturating_mul(up)
    };
    scaled.clamp(CHUNK_MIN, CHUNK_MAX)
}

/// Search for a valid PoW nonce using the current Rayon pool.
/// Internal miner calls this inside the process-wide all-core PoW phase.
///
/// Detached witness fields are not part of the PoW hash; only semantic header
/// fields are absorbed under the `POWHDR__` domain.
///
/// Returns `Some(PowSolution)` when found, or `None` if cancelled via the
/// `cancel` channel (when a new P2P block arrives or a new template is ready).
pub fn search_pow_parallel(
    header_template: &BlockHeader,
    cancel: &std::sync::atomic::AtomicBool,
) -> Option<PowSolution> {
    // Start from a random nonce to avoid all miners/restarts colliding on nonce=0.
    // Uses a simple time-based seed — not cryptographic, just for nonce diversity.
    let random_start: u128 = {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u128;
        // Mix with block height for additional diversity.
        (t ^ (header_template.height as u128).wrapping_mul(0x9E3779B97F4A7C15))
            & 0xFFFF_FFFF_FFFF_FFFF // 64-bit random start
    };
    search_pow_parallel_from(header_template, cancel, random_start)
}

/// Testable twin of [`search_pow_parallel`] with the first nonce injected.
///
/// Production always draws its own start; this exists so a test can place a
/// known solution inside one thread's share and watch how long it takes to
/// surface. Nothing else differs.
fn search_pow_parallel_from(
    header_template: &BlockHeader,
    cancel: &std::sync::atomic::AtomicBool,
    first_nonce: u128,
) -> Option<PowSolution> {
    use rayon::prelude::*;
    use std::sync::atomic::Ordering;

    let random_start = first_nonce;

    // Partition the 128-bit nonce space into thread-sized chunks.
    // Each thread checks cancel once per nonce batch; this keeps cancellation
    // responsive without paying an atomic load for every permutation.
    const DIGEST_BATCH: usize = 256;
    /// Batch size once the cache-resident walk is armed.
    ///
    /// A batch is the cancellation granularity: the thread checks `cancel` once
    /// per batch. At 21.8 µs per sponge digest, 256 of them is 5.6 ms and nobody
    /// notices. At 1.92 ms per walked digest, 256 would be **half a second** of
    /// deafness — long enough for a miner to keep grinding a template whose parent
    /// has already been replaced. Four keeps it at about 8 ms, the same order as
    /// today, and the batch buys nothing anyway: the walk dominates the sponge by
    /// a factor of eighty-eight, so there is no amortisation left to chase.
    ///
    /// Both figures are the **unit cost on one idle EPYC 7742 core** [MEASURED
    /// 2026-09-23; `jetsam_chain::consensus::pow::pow_digest` is the table of
    /// record]. A thread on a saturated machine sees 4.66 ms instead, which only
    /// makes this batch size more right, never less.
    const WALK_BATCH: usize = 4;
    let target = header_template.difficulty_target;
    // The digest a block at this height must satisfy. `None` on every profile
    // today, so this whole path is byte-for-byte what it was.
    let walk = jetsam_chain::consensus::params::v1_4_active(header_template.height);

    let mut start_nonce: u128 = random_start;

    // Hoist thread count — it never changes during a chunk.
    let num_threads = rayon::current_num_threads();
    // Sized from the previous pass, so a pass costs about `CHUNK_TARGET`
    // whatever the machine. A thread's share — and therefore how long it keeps
    // grinding a parent the network has already replaced — follows from this.
    let mut chunk_size = CHUNK_START;
    let mut per_thread = chunk_size.div_ceil(num_threads as u128);
    let fields = pow_header_fields(header_template);

    // Measures only the time actually spent searching, so the reported rate is
    // this machine's TowerHash speed and not an average diluted by proving.
    let mut ticker = SearchTicker::start();

    // Raised by the thread that wins, so the others stop within a batch instead
    // of grinding their whole share out.
    //
    // `find_map_any` does not interrupt a closure that is already running: every
    // thread finishes its `CHUNK_SIZE / threads` nonces before the solution is
    // handed back. Below the walk a share cost 0.15 s and nobody noticed. At
    // 4.66 ms per walked digest the same share is 31 s — [MEASURED 2026-09-23 on
    // the test chain at the v1.4 crossing: per-thread throughput with every core
    // busy, 2.5x the 1.85 ms unit cost of one digest on an idle core], where
    // `pow_ms` collapsed onto multiples of one chunk and every block was announced
    // that long after it had been won. Every one of those seconds goes to whoever
    // else is mining the same parent. The saturated figure is the right one here:
    // this is about what a thread in the pool actually waits.
    //
    // The external miner has carried this flag since before the fork; this is
    // the same fix on the stage that had been left without it.
    let found = std::sync::atomic::AtomicBool::new(false);

    loop {
        if cancel.load(Ordering::Relaxed) || crate::cpu_budget::pow_preemption_requested() {
            report_rate(&mut ticker, header_template.height, num_threads);
            return None;
        }

        // Search this chunk in parallel. Each thread reuses one canonical field
        // schedule and computes nonce digests in packed Poseidon2b batches.
        let pass_started = Instant::now();
        let chunk_end = start_nonce + chunk_size;
        let solution: Option<PowSolution> =
            (0..num_threads).into_par_iter().find_map_any(|thread_id| {
                let thread_start = start_nonce + (thread_id as u128) * per_thread;
                let thread_end = (thread_start + per_thread).min(chunk_end);
                if thread_start >= thread_end {
                    return None;
                }
                let fields = fields;
                let mut hasher = PowNonceBatchHasher::new(&fields);
                let mut digests = [[0u8; 32]; DIGEST_BATCH];
                // One scratchpad per thread, allocated once for the whole chunk.
                // 512 KiB per nonce would cost more than the walk itself.
                let mut scratch = walk.then(jetsam_chain::consensus::pow_walk::Scratch::new);
                let batch = if walk { WALK_BATCH } else { DIGEST_BATCH };
                let mut nonce = thread_start;

                while nonce < thread_end {
                    if found.load(Ordering::Relaxed)
                        || cancel.load(Ordering::Relaxed)
                        || crate::cpu_budget::pow_preemption_requested()
                    {
                        return None;
                    }
                    let n = ((thread_end - nonce).min(batch as u128)) as usize;
                    hasher.hash_into(nonce, &mut digests[..n]);
                    POW_METER.add_hashes(n as u64);
                    for (i, seed) in digests[..n].iter().enumerate() {
                        let hash = match scratch.as_mut() {
                            Some(sc) => {
                                jetsam_chain::consensus::pow_walk::towerwalk_digest_with(sc, seed)
                            }
                            None => *seed,
                        };
                        if le256_lt(&hash, &target) {
                            found.store(true, Ordering::Relaxed);
                            return Some(PowSolution {
                                nonce: nonce + i as u128,
                                pow_hash: hash,
                            });
                        }
                    }
                    nonce += n as u128;
                }
                None
            });

        if solution.is_some() {
            report_rate(&mut ticker, header_template.height, num_threads);
            return solution;
        }

        // Check cancellation before next chunk.
        if cancel.load(Ordering::Relaxed) || crate::cpu_budget::pow_preemption_requested() {
            report_rate(&mut ticker, header_template.height, num_threads);
            return None;
        }

        report_rate(&mut ticker, header_template.height, num_threads);

        // Re-size the next pass from what this one actually cost. Done here,
        // after the cancellation checks, so a pass that ended early on a moved
        // tip does not teach the miner that it is faster than it is.
        chunk_size = next_chunk_size(chunk_size, pass_started.elapsed());
        per_thread = chunk_size.div_ceil(num_threads as u128);

        // Advance to the next chunk.
        start_nonce = start_nonce.saturating_add(chunk_size);
        if start_nonce == 0 {
            // Nonce space exhausted (extremely unlikely with 128-bit nonce).
            return None;
        }
    }
}

#[cfg(test)]
mod pow_search_tests {
    use super::*;
    use jetsam_chain::block_header::BlockHeader;
    use jetsam_poseidon2b::primitives::Address;
    use std::sync::atomic::AtomicBool;

    /// About one nonce in 2^24 satisfies this.
    const TEST_TARGET: [u8; 32] = {
        let mut t = [0u8; 32];
        t[29] = 0x01; // hash < 2^232
        t
    };

    /// Found once, offline, against the header below. Re-derive it with the
    /// same header if either ever changes — the target is part of the hashed
    /// field schedule, so a different target is a different solution.
    const KNOWN_SOLUTION: u128 = 24_491_018;

    fn test_header() -> BlockHeader {
        BlockHeader {
            prev_block_hash: [0u8; 32],
            state_root: [1u8; 32],
            tx_root: [2u8; 32],
            timestamp: 1_700_000_000,
            height: 1,
            miner_address: Address([3u8; 32]),
            nonce: 0,
            difficulty_target: TEST_TARGET,
            log_slots: 24,
            active_slot_count: 0,
            alloc_counter: 0,
        }
    }

    #[test]
    fn a_solution_surfaces_as_soon_as_one_thread_finds_it() {
        // Two threads split the chunk into halves of 500 000 nonces. The first
        // half starts one nonce before the known solution, so its thread finds
        // it inside the first batch. One nonce in 2^24 meets this target, so the
        // other half almost certainly holds nothing its thread can reach: left
        // alone it grinds all 500 000 before returning None, and `find_map_any`
        // does not interrupt a closure already running.
        //
        // Before the cache-resident walk a share cost 0.15 s and nobody noticed.
        // At 4.66 ms per walked digest the same share is 31 s — [MEASURED
        // 2026-09-23 on the test chain at the v1.4 crossing: per-thread
        // throughput with every core busy, 2.5x the 1.85 ms unit cost of one
        // digest on an idle core], where every block was announced that long
        // after it had been won, and every one of those seconds is handed to a
        // competitor mining the same parent.
        let _serialised = METER_OBSERVERS.lock().unwrap_or_else(|e| e.into_inner());
        let header = test_header();
        let cancel = AtomicBool::new(false);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .expect("two-thread pool");

        let started = Instant::now();
        let solution =
            pool.install(|| search_pow_parallel_from(&header, &cancel, KNOWN_SOLUTION - 1));
        let elapsed = started.elapsed();

        let solution = solution.expect("the known solution sits in the first thread's share");
        assert_eq!(solution.nonce, KNOWN_SOLUTION);
        assert!(le256_lt(&solution.pow_hash, &TEST_TARGET));
        assert!(
            elapsed < Duration::from_secs(10),
            "a won block waited {elapsed:?} on threads that had already lost"
        );
    }

    /// The pass has to shrink until it costs about `CHUNK_TARGET`, whatever
    /// the machine — that duration is how long a thread keeps grinding a
    /// parent the network has already replaced.
    #[test]
    fn a_pass_converges_on_its_target_duration() {
        for hps in [740.0_f64, 2_960.0, 50_000.0, 5_000_000.0] {
            let mut size = super::CHUNK_SIZE;
            for _ in 0..24 {
                let took = Duration::from_secs_f64(size as f64 / hps);
                size = super::next_chunk_size(size, took);
            }
            let settled = Duration::from_secs_f64(size as f64 / hps);
            assert!(
                settled <= super::CHUNK_TARGET * 2 || size == super::CHUNK_MIN,
                "at {hps} H/s a pass still costs {settled:?} (size {size})"
            );
        }
    }

    /// The case this exists for. A four-thread miner on the test chain ran at
    /// 2.96 kH/s; one pass of 1,000,000 nonces took it five and a half minutes,
    /// on a chain that produces a block every 90 seconds. It could not win.
    #[test]
    fn a_small_miner_reacts_faster_than_the_block_interval() {
        const MEASURED_HPS: f64 = 2_960.0;
        const BLOCK_INTERVAL: Duration = Duration::from_secs(90);

        let before = Duration::from_secs_f64(super::CHUNK_SIZE as f64 / MEASURED_HPS);
        assert!(
            before > BLOCK_INTERVAL,
            "the regression this guards no longer reproduces: {before:?}"
        );

        let mut size = super::CHUNK_SIZE;
        for _ in 0..24 {
            let took = Duration::from_secs_f64(size as f64 / MEASURED_HPS);
            size = super::next_chunk_size(size, took);
        }
        let after = Duration::from_secs_f64(size as f64 / MEASURED_HPS);
        assert!(
            after * 10 < BLOCK_INTERVAL,
            "a four-thread miner still spends {after:?} per pass against a \
             {BLOCK_INTERVAL:?} block interval"
        );
    }

    /// The first pass is the one that decides whether a new miner joins the
    /// race at all. Starting it at 1,000,000 nonces cost the measured
    /// four-thread machine five and a half minutes before it could even notice
    /// the tip had moved — adapting from the second pass onward was too late.
    #[test]
    fn the_very_first_pass_is_short_on_a_slow_machine() {
        const MEASURED_HPS: f64 = 2_960.0;
        let first = Duration::from_secs_f64(super::CHUNK_START as f64 / MEASURED_HPS);
        assert!(
            first <= Duration::from_secs(1),
            "the first pass already costs {first:?} before anything is measured"
        );
        let old = Duration::from_secs_f64(super::CHUNK_SIZE as f64 / MEASURED_HPS);
        assert!(
            old > Duration::from_secs(60),
            "the regression this guards no longer reproduces: {old:?}"
        );
    }

    /// One slow pass must not collapse the size, and one fast pass must not
    /// blow it up: a machine shared with other work would oscillate.
    #[test]
    fn a_single_odd_pass_moves_the_size_by_at_most_four() {
        let size = 100_000_u128;
        assert_eq!(super::next_chunk_size(size, super::CHUNK_TARGET * 1000), size / 4);
        assert_eq!(super::next_chunk_size(size, super::CHUNK_TARGET / 1000), size * 4);
        assert_eq!(super::next_chunk_size(size, super::CHUNK_TARGET), size);
    }

    /// Never zero, never unbounded — a zero pass would spin without hashing.
    #[test]
    fn the_size_stays_inside_its_floor_and_ceiling() {
        assert_eq!(super::next_chunk_size(1, Duration::from_secs(3600)), super::CHUNK_MIN);
        assert_eq!(
            super::next_chunk_size(u128::MAX, Duration::from_nanos(1)),
            super::CHUNK_MAX
        );
        assert!(super::next_chunk_size(super::CHUNK_MIN, Duration::from_secs(9)) >= super::CHUNK_MIN);
    }
}
