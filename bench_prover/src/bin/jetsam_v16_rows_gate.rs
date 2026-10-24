// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Rows gate of the v1.6 relation: the useful rows of the B24 and B255 classes of
//! the v1.6 relation (the batch slot in place of the v1.5 client slot), on
//! honest witnesses, beside the v1.5 relation built from the same witnesses.
//! Passes when each v1.6 class fits its v1.5 dimension (2^23 / 2^25) and adds
//! at most 0.5 M rows to v1.5. No proof, no pack written.
//!
//! Usage: `jetsam_v16_rows_gate` (release; the profile is the build's).

use bench_prover::HonestHistoryStepFixtureProvider;
use jetsam_chain::consensus::params::HistoryStepPackGeneration;

/// The pack generator's seed (`jetsam_matrix_gen`): the same honest witnesses.
const FIXTURE_SEED: u128 = 0x4849_5354_4550_5f56_31;

fn main() {
    let started = std::time::Instant::now();
    // The v1.5 block rules: v1.6 does not move them (same ladder, boundary,
    // anchors), so the same honest children feed both relations.
    let mut provider = HonestHistoryStepFixtureProvider::new_in(FIXTURE_SEED, HistoryStepPackGeneration::V1_5)
        .unwrap_or_else(|error| panic!("honest fixtures: {error}"));
    let gate = jetsam_recursive::measure_history_step_v1_6_rows(&mut provider)
        .unwrap_or_else(|error| panic!("v1.6 rows gate: {error}"));
    println!("v1.6 rows gate ({:.1} s)", started.elapsed().as_secs_f64());
    println!(
        "  Link key w_log (L-A, L-B, L-C): v1.5 {:?}, v1.6 {:?}",
        gate.v1_5_link_w_logs, gate.v1_6_link_w_logs
    );
    let mut pass = true;
    for (slot, name) in ["B24", "B255"].iter().enumerate() {
        let v15 = gate.v1_5_rows[slot];
        let v16 = gate.v1_6_rows[slot];
        let limit = gate.v1_6_limits[slot];
        let delta = v16 as i64 - v15 as i64;
        let ok = v16 <= limit && delta <= 500_000;
        pass &= ok;
        println!(
            "  {name}: v1.5 {v15} rows, v1.6 {v16} rows, delta {delta:+}, class 2^{} = {limit} ({:.2} % used) -> {}",
            limit.trailing_zeros(),
            100.0 * v16 as f64 / limit as f64,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!("GATE E5: {}", if pass { "PASS" } else { "FAIL" });
    assert!(pass, "gate E5 failed");
}
