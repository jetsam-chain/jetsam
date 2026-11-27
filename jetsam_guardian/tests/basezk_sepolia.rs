// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Read only: Base Sepolia's real dispute game factory on the L1 Sepolia, through a public endpoint, read by the
//! functions `jg-basezk` reads with (`basezk::probe`): the factory's current keys, its newest games, each one's type,
//! L2 block, status, `proofCount`, `zkProver()` and `expectedResolution`. No transaction, no key, nothing signed.
//!
//! Ignored by default (it needs the network):
//! `cargo test -p jetsam_guardian --test basezk_sepolia -- --ignored --nocapture`;
//! the L1 Sepolia endpoint in `JETSAM_BASEZK_L1` (required: an `https://` or `http://` JSON-RPC URL), the number of
//! newest games in `JETSAM_BASEZK_LAST` (default 24).

use jetsam_chain::consensus::destination::base_read::BASE_SEPOLIA;
use jetsam_guardian::basezk::{probe, Hx, KeySet};
use jetsam_guardian::l1::L1;

/// The keys of Base Sepolia's type-621 implementation as read on 2026-11-10 (RETOUR24H_ETUDE §2.1, RET24/01).
fn measured_2026_11_10() -> KeySet {
    let h20 = |s: &str| Hx::<20>(hex::decode(s).unwrap().try_into().unwrap());
    let h32 = |s: &str| Hx::<32>(hex::decode(s).unwrap().try_into().unwrap());
    KeySet {
        game_impl: h20("4234216f1648a8119c6e41e6c0c73956ecc850af"),
        zk_range_hash: h32("3d3713d31a8936563bc56791040067131bb3fa3c0ab8b5e65e9c4cc0755b5fdb"),
        zk_aggregate_hash: h32("00d5a43068d7a9d1eac440a790b776c749ae6516c43f1a572226190cef6fd45a"),
        config_hash: h32("12e9c45f19f9817c6d4385fad29e7a70c355502cf0883e76a9a7e478a85d1360"),
        // AggregateVerifier 0.3.0's FAST_FINALIZATION_DELAY (1 day), read with the keys since B6
        fast_finalization_delay: 86_400,
    }
}

#[test]
#[ignore = "reads Base Sepolia's factory on a public L1 Sepolia endpoint (network)"]
fn base_sepolia_games_their_proofs_and_their_keys() {
    let url = std::env::var("JETSAM_BASEZK_L1").expect("JETSAM_BASEZK_L1: the L1 Sepolia JSON-RPC endpoint to read");
    let last: u64 = std::env::var("JETSAM_BASEZK_LAST").ok().and_then(|v| v.parse().ok()).unwrap_or(24);
    let l1 = L1::new(&url);
    let p = probe(&l1, &BASE_SEPOLIA, last, "finalized").expect("the probe reads the factory");
    println!("endpoint {url}");
    println!("L1 Sepolia finalized block {} (time {}), factory 0x{}: gameCount {}", p.l1_block, p.l1_time, hex::encode(BASE_SEPOLIA.dispute_game_factory), p.count);
    println!("factory's type-621 keys now: {}", p.factory_keys);
    if p.factory_keys == measured_2026_11_10() {
        println!("  = the keys measured on 2026-11-10 (no rotation since)");
    } else {
        println!("  ≠ the keys measured on 2026-11-10: {} (a rotation since, K4)", measured_2026_11_10());
    }
    let (mut with_zk, mut tee_only) = (0, 0);
    for r in &p.rows {
        match (&r.live, &r.keys) {
            (Some(live), keys) => {
                let age_h = p.l1_time.saturating_sub(r.created_at) as f64 / 3_600.0;
                let window_h = live.expected_resolution.saturating_sub(r.created_at) as f64 / 3_600.0;
                println!(
                    "  #{} type {} game {} L2 block {} created {:.1} h ago, status {}, proofCount {}, zkProver {}, expectedResolution created + {:.2} h, keys {}",
                    r.index,
                    r.game_type,
                    r.game,
                    r.l2_block.unwrap_or(0),
                    age_h,
                    live.status,
                    live.proof_count,
                    live.zk_prover,
                    window_h,
                    match keys {
                        Some(k) if *k == p.factory_keys => "= the factory's".to_string(),
                        Some(k) => format!("{k}"),
                        None => "none (not an exact clone)".to_string(),
                    }
                );
                if live.zk_prover.is_zero() {
                    tee_only += 1;
                } else {
                    with_zk += 1;
                }
            }
            (None, _) => println!("  #{} type {} game {}: another type, skipped", r.index, r.game_type, r.game),
        }
    }
    println!("{} games of type 621 read: {with_zk} with a ZK proof, {tee_only} with the TEE proof only", with_zk + tee_only);
    // what holds whatever Base does next
    assert_eq!(p.rows.len() as u64, last.min(p.count));
    assert!(p.count > 30_000, "Base Sepolia's factory counted 30 052 games on 2026-11-10");
    let games: Vec<_> = p.rows.iter().filter(|r| r.game_type == 621).collect();
    assert!(!games.is_empty(), "Base Sepolia proposes a type-621 game every ≈ 20 min");
    for r in games {
        let live = r.live.expect("a type-621 game is read");
        assert!(r.keys.is_some(), "#{}: every game is an exact clone of an implementation", r.index);
        assert!(r.l2_block.unwrap_or(0) > 50_000_000, "#{}: an L2 block of Base Sepolia", r.index);
        assert!(live.proof_count >= 1, "#{}: a game is created with its TEE proof", r.index);
        if live.zk_prover.is_zero() && live.proof_count == 1 && live.status == 0 {
            assert_eq!(live.expected_resolution - r.created_at, 5 * 86_400, "#{}: one proof, 5 days", r.index);
        }
    }
}
