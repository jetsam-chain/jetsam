// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-forge-equivocation` — **TEST ONLY** (private rehearsal, T7.5/T7.6): plays a thief who holds 5
//! TEST shares (the T0.5 TEST dealing `fixture`) OUTSIDE every signer and its journal: it signs a
//! statement that contradicts an honest group signature of the archive (same height, another block
//! id: rule E1) and publishes it (`<point>/forged/`), as a leaked signature would be. It also writes the
//! matching proposal (`--proposal`) to show that the honest signers refuse to sign it.
//!   jg-forge-equivocation --archive <signatures.jsonl> --out <forged dir> [--proposal <proposals dir>]

use jetsam_guardian::checkpoint::Checkpoint;
use jetsam_guardian::threshold::{combine, g2_eip2537, partial_sign, test_dealing, verify};
use jetsam_guardian::GuardianStatement;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let get = |n: &str| a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned();
    let archive = std::fs::read_to_string(get("--archive").expect("--archive"))?;
    let out = std::path::PathBuf::from(get("--out").expect("--out"));
    // the newest honest co-signature of the archive
    let entry: serde_json::Value = archive
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .filter(|e| e["kind"] == "cosign")
        .last()
        .ok_or_else(|| anyhow::anyhow!("no co-signature in the archive"))?;
    let honest: [u8; 416] = hex::decode(entry["group"]["statement"].as_str().unwrap_or_default())?.try_into().map_err(|_| anyhow::anyhow!("416"))?;
    let mut s = GuardianStatement::decode(&honest).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    s.block_id = jetsam_guardian::keccak256(b"jetsam-v17-rehearsal/forged-block (TEST)");
    let forged = s.encode();
    let m = jetsam_guardian::keccak256(&forged);
    let d = test_dealing("fixture");
    let partials: Vec<(u32, [u8; 96])> = d.shares[..5].iter().map(|sh| (sh.index, partial_sign(sh, &m))).collect();
    let sig = combine(&partials).map_err(|e| anyhow::anyhow!("{e}"))?;
    anyhow::ensure!(verify(&sig, &m, &d.group_key), "the forged signature verifies under the group key");
    std::fs::create_dir_all(&out)?;
    let v = serde_json::json!({"_note": "TEST: forged with 5 TEST shares outside the signers", "height": s.height,
        "statement": hex::encode(forged), "sig96": hex::encode(sig), "sig256": hex::encode(g2_eip2537(&sig).map_err(|e| anyhow::anyhow!("{e}"))?),
        "contradicts": entry["group"]["m"]});
    std::fs::write(out.join(format!("forged-{}.json", s.height)), v.to_string())?;
    if let Some(dir) = get("--proposal") {
        let p = entry["proposal"].clone();
        let mut cp: Checkpoint = serde_json::from_value(p["checkpoint"].clone())?;
        cp.locked_block_id = s.block_id;
        let mut q = p.clone();
        q["checkpoint"] = serde_json::to_value(cp)?;
        q["kind"] = "cosign".into();
        std::fs::write(std::path::Path::new(&dir).join("zz-forged.json"), q.to_string())?;
    }
    println!("{v}");
    Ok(())
}
