// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! M2 task 2.6 — a real client of the imposed form, end to end outside the
//! block: catalogue entry 1 (`jetsam_client_agent`: an AI agent's trace
//! respected its tool policy and its budget), built in the canonical client
//! form (m = k_log = 22, rate 1/4, batch 2^5), proved with
//! `jetsam_ivc_prover`, received by a miner (the native pre-pass), and its
//! published lanes checked by a node against the chain's registry. A trace
//! that violates the policy is shown to have no valid proof.
//!
//! ```text
//! CARGO_TARGET_DIR=... RAYON_NUM_THREADS=32 nice -n 10 \
//!   cargo bench -p bench_prover --features client-slot --bench client_agent_demo
//! ```
//! `DEMO_SAMPLES` (default 2) repeats the prove / verify timings.

use std::sync::Arc;
use std::time::Instant;

use jetsam_client_agent::{
    agent_policy_instance, complies, AgentPolicy, ToolCall, MAX_CALLS,
};
use jetsam_ivc_core::challenger::FsLaneChallenger;
use jetsam_ivc_core::field::F128;
use jetsam_ivc_core::matrix_claim::c1::fresh_claim_value_c1;
use jetsam_ivc_core::verifier::verify_field_c1_deferred_matrix_with_post_commit_context;
use jetsam_recursive::{
    encode_history_step_client_proof, history_step_bank_io_layout_with_client,
    history_step_client_proof_max_wire_bytes, parse_history_step_client_lanes,
    HistoryStepChainClients, HistoryStepClientForm, HistoryStepClientRegistry,
    HistoryStepClientWitness, PreparedHistoryStepClient, HISTORY_STEP_CLIENT_PROOF_DOMAIN,
};
use jetsam_chain::consensus::params::HistoryStepPackGeneration;

fn status_kib(field: &str) -> i64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status.lines().find_map(|line| {
                line.strip_prefix(field)
                    .and_then(|rest| rest.trim().trim_end_matches("kB").trim().parse().ok())
            })
        })
        .unwrap_or(-1)
}

/// Reset the resident high-water mark (`VmHWM`) to the current RSS.
fn reset_hwm() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

fn gib(kib: i64) -> f64 {
    kib as f64 / (1024.0 * 1024.0)
}

fn load(label: &str) {
    let loadavg = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
    println!("[demo-load] {label}: loadavg {}", loadavg.trim());
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1e3
}

fn policy() -> AgentPolicy {
    AgentPolicy {
        // search, fetch, summarize, code-run, calendar-read, mail-draft, db-read
        allowed_tools: [101, 102, 103, 205, 206, 300, 777],
        budget_cap: 250_000,
    }
}

fn call(tool: u32, cost: u32, receipt: u8) -> ToolCall {
    ToolCall {
        tool,
        cost,
        receipt_hash: [receipt; 32],
    }
}

/// Twelve calls, 187 400 spent of 250 000.
fn honest_trace() -> Vec<ToolCall> {
    [
        (101, 1_200),
        (102, 3_400),
        (103, 25_000),
        (205, 90_000),
        (101, 1_200),
        (102, 3_400),
        (206, 700),
        (300, 12_000),
        (777, 500),
        (103, 25_000),
        (300, 12_000),
        (102, 13_000),
    ]
    .iter()
    .enumerate()
    .map(|(index, (tool, cost))| call(*tool, *cost, index as u8 + 1))
    .collect()
}

fn prove(
    form: &HistoryStepClientForm,
    r1cs: &jetsam_ivc_core::field_r1cs::FieldR1cs,
    witness: &[F128],
    io: &[F128],
) -> (
    jetsam_ivc_core::proof::C1FieldR1csProof,
    jetsam_ivc_core::pcs::Commitment,
) {
    let mut prover = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
    let (proof, (), commitment, _) =
        jetsam_ivc_prover::field_prover::prove_field_c1_with_public_io_and_post_commit_context(
            r1cs,
            witness,
            form.pcs_params(),
            form.io_spec(),
            io,
            &form.post_commit_digest(),
            &mut prover,
            |_| (),
        );
    (proof, commitment)
}

fn run() -> Result<(), String> {
    let samples: usize = std::env::var("DEMO_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(2);
    println!(
        "[demo] rayon threads={} samples={samples} max_calls={MAX_CALLS}",
        rayon::current_num_threads()
    );
    let form = HistoryStepClientForm::canonical();
    let shape = form.shape();
    let slice = form.io_spec().io_slice;
    let calls = honest_trace();
    assert!(complies(&policy(), &calls));

    // 1. The circuit in the imposed form, and its registered digest D.
    load("build");
    let started = Instant::now();
    let instance = agent_policy_instance(&policy(), &calls, shape, slice)
        .map_err(|error| format!("instance: {error}"))?;
    let build_ms = ms(started);
    let started = Instant::now();
    let digest = instance.r1cs.structural_statement_digest();
    let digest_ms = ms(started);
    println!(
        "[demo] circuit: {} rows used of 2^{} ({:.2} %), built in {build_ms:.0} ms; \
         D={} hashed in {digest_ms:.0} ms",
        instance.circuit_rows,
        shape.m,
        100.0 * instance.circuit_rows as f64 / (1u64 << shape.m) as f64,
        hex::encode(digest)
    );
    let started = Instant::now();
    let satisfied = instance.r1cs.satisfies(&instance.witness);
    println!("[demo] honest witness satisfies D: {satisfied} ({:.0} ms)", ms(started));
    if !satisfied {
        return Err("honest witness does not satisfy D".into());
    }
    let matrix = Arc::new(instance.r1cs);

    // 2. The client proves (the agent's operator, off chain).
    let mut proved = None;
    for sample in 0..samples {
        load(&format!("prove sample {}", sample + 1));
        reset_hwm();
        let started = Instant::now();
        let (proof, commitment) = prove(&form, &matrix, &instance.witness, &instance.io);
        let prove_ms = ms(started);
        println!(
            "[demo] prove sample {}: {prove_ms:.0} ms, VmHWM(phase)={:.2} GiB",
            sample + 1,
            gib(status_kib("VmHWM:"))
        );
        proved = Some((proof, commitment));
    }
    let (proof, commitment) = proved.ok_or("no sample")?;
    let bytes = encode_history_step_client_proof(&form, &proof, &commitment.root, &instance.io)
        .map_err(|error| format!("encode: {error}"))?;
    println!(
        "[demo] proof on the wire: {} bytes (shared Merkle paths); fixed unshared bound {} bytes",
        bytes.len(),
        history_step_client_proof_max_wire_bytes(&form).map_err(|error| error.to_string())?
    );

    // 3. Verification: the native C1 verifier (lincheck deferred), then the
    //    lincheck closed against the registered matrix.
    for sample in 0..samples {
        load(&format!("verify sample {}", sample + 1));
        let started = Instant::now();
        let mut challenger = FsLaneChallenger::new_c1(HISTORY_STEP_CLIENT_PROOF_DOMAIN);
        let (_, fresh) = verify_field_c1_deferred_matrix_with_post_commit_context(
            &shape,
            &digest,
            &commitment,
            &proof,
            form.io_spec(),
            &instance.io,
            &form.post_commit_digest(),
            &(),
            &mut challenger,
            |_, _| Ok(()),
        )
        .map_err(|error| format!("verify: {error:?}"))?;
        let deferred_ms = ms(started);
        let started = Instant::now();
        let holds = fresh_claim_value_c1(&matrix, &fresh) == fresh.value;
        let lincheck_ms = ms(started);
        println!(
            "[demo] verify sample {}: deferred verifier {deferred_ms:.0} ms + lincheck on D \
             {lincheck_ms:.0} ms = {:.0} ms; lincheck holds: {holds}",
            sample + 1,
            deferred_ms + lincheck_ms
        );
        if !holds {
            return Err("lincheck does not hold on D".into());
        }
    }

    // 4. The chain's registry (static in the prototype) and a miner's
    //    reception pre-pass (native verification + lincheck + fold + replay).
    let started = Instant::now();
    let registry = HistoryStepClientRegistry::new(form.registry_depth(), vec![digest])
        .map_err(|error| error.to_string())?;
    let chain = HistoryStepChainClients::new(&form, registry.clone(), vec![matrix.clone()])
        .map_err(|error| error.to_string())?;
    println!(
        "[demo] chain registry installed (1 of {} entries, root {}) in {:.0} ms",
        form.registry_capacity(),
        hex::encode(chain.root()),
        ms(started)
    );
    let witness = HistoryStepClientWitness {
        field_proof: proof,
        commitment,
        io: instance.io.clone(),
        matrix: matrix.clone(),
        registry,
    };
    let mut prepared = None;
    for sample in 0..samples {
        load(&format!("reception sample {}", sample + 1));
        let started = Instant::now();
        prepared = Some(
            PreparedHistoryStepClient::prepare(&form, &witness)
                .map_err(|error| format!("reception: {error}"))?,
        );
        println!("[demo] reception pre-pass sample {}: {:.0} ms", sample + 1, ms(started));
    }
    let prepared = prepared.ok_or("no sample")?;

    // 5. The node: the block's client lanes, parsed and checked.
    let lanes = history_step_bank_io_layout_with_client(HistoryStepPackGeneration::V1_3)
        .client
        .ok_or("client lanes")?;
    let mut block_io = vec![F128::ZERO; lanes.end()];
    prepared
        .install_lanes(&lanes, &mut block_io)
        .map_err(|error| error.to_string())?;
    for sample in 0..samples {
        let started = Instant::now();
        let claim = parse_history_step_client_lanes(&lanes, &block_io)
            .map_err(|error| error.to_string())?
            .ok_or("absent client")?;
        chain
            .check_claim(&claim)
            .map_err(|error| format!("node check: {error}"))?;
        println!("[demo] node check of the client lane, sample {}: accepted in {:.0} ms", sample + 1, ms(started));
    }

    // 6. A trace that violates the policy: same D, no satisfying witness,
    //    and a proof forced out of the prover anyway is refused on reception.
    for (name, violating) in [
        ("forbidden tool 666", {
            let mut calls = honest_trace();
            calls[3].tool = 666;
            calls
        }),
        ("budget 250 001 > 250 000", {
            let mut calls = honest_trace();
            calls[11].cost = 13_000 + 62_601;
            calls
        }),
    ] {
        assert!(!complies(&policy(), &violating));
        let instance = agent_policy_instance(&policy(), &violating, shape, slice)
            .map_err(|error| format!("{name}: {error}"))?;
        let same_d = instance.r1cs.structural_statement_digest() == digest;
        let satisfied = instance.r1cs.satisfies(&instance.witness);
        let forced = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            prove(&form, &instance.r1cs, &instance.witness, &instance.io)
        }));
        let outcome = match forced {
            Err(_) => "the prover panicked (no proof)".to_string(),
            Ok((proof, commitment)) => {
                let forced = HistoryStepClientWitness {
                    field_proof: proof,
                    commitment,
                    io: instance.io.clone(),
                    matrix: matrix.clone(),
                    registry: witness.registry.clone(),
                };
                match PreparedHistoryStepClient::prepare(&form, &forced) {
                    Ok(_) => return Err(format!("{name}: a violating trace was accepted")),
                    Err(error) => format!("forced proof refused on reception: {error}"),
                }
            }
        };
        println!("[demo] violating trace ({name}): same D={same_d}, witness satisfies D={satisfied}; {outcome}");
        if !same_d || satisfied {
            return Err(format!("{name}: violating trace not refused by D"));
        }
    }
    println!("[demo] done");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("[demo] FAILED: {error}");
        std::process::exit(1);
    }
}
