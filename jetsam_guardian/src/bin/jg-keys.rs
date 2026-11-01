// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-keys` — the keys of ONE guardian machine (design §6.1, §6.2 step 10):
//!
//! * `machine-init --state <dir> --generation <g> --passphrase-fd <n> [--test-kdf]`:
//!   a new X25519 machine key, sealed by the operator's passphrase
//!   (`<state>/sign/machine-g<g>.key.sealed`); prints its public key and
//!   fingerprint (to compare on paper);
//! * `feed-init --state <dir>`: the Ed25519 feed key of `jg-net`
//!   (`<state>/net/feed.key`), prints its public key;
//! * `import --config <guardian.toml> --sealed <share file> --passphrase-fd <n>
//!   --h0-header <hex>`: opens the share with the machine key, checks it
//!   against the Feldman commitments, checks a partial on the ceremony's test
//!   message against `pk_i`, installs the sealed share and writes the
//!   journal's GENESIS record with the trust anchor H0 (the genesis header
//!   given, W(H0) recomputed from it, its id equal to the network).

use std::io::Read;
use std::os::unix::io::FromRawFd;
use std::path::PathBuf;

use jetsam_guardian::ceremony::test_message;
use jetsam_guardian::chain::HeaderInfo;
use jetsam_guardian::config::GuardianConfig;
use jetsam_guardian::feed::FeedKey;
use jetsam_guardian::journal::{GenesisBody, Journal};
use jetsam_guardian::seal::{open_machine_key, open_sealed, seal_machine_key, share_aad, KdfCost, MachineKey};
use jetsam_guardian::threshold::{partial_sign, share_public_key, verify, verify_share, Share};

fn passphrase(fd: i32) -> anyhow::Result<zeroize::Zeroizing<Vec<u8>>> {
    let mut f = unsafe { std::fs::File::from_raw_fd(fd) };
    let mut buf = zeroize::Zeroizing::new(Vec::new());
    f.read_to_end(&mut buf)?;
    while buf.last().is_some_and(|b| *b == b'\n') {
        buf.pop();
    }
    anyhow::ensure!(!buf.is_empty(), "empty passphrase");
    Ok(buf)
}

fn run() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let get = |n: &str| a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).cloned().ok_or_else(|| anyhow::anyhow!("missing {n}"));
    match a.get(1).map(String::as_str) {
        Some("machine-init") => {
            let state = PathBuf::from(get("--state")?);
            let gen: u32 = get("--generation")?.parse()?;
            let cost = if a.iter().any(|x| x == "--test-kdf") { KdfCost::TEST } else { KdfCost::PRODUCTION };
            let pass = passphrase(get("--passphrase-fd")?.parse()?)?;
            let path = state.join("sign").join(format!("machine-g{gen}.key.sealed"));
            anyhow::ensure!(!path.exists(), "{} exists: never overwritten", path.display());
            std::fs::create_dir_all(path.parent().expect("dir"))?;
            let k = MachineKey::generate()?;
            std::fs::write(&path, seal_machine_key(&k, &pass, cost)?)?;
            println!("{{\"machine_public\":\"{}\",\"fingerprint\":\"{}\"}}", hex::encode(k.public), k.fingerprint());
        }
        Some("feed-init") => {
            let state = PathBuf::from(get("--state")?);
            let path = state.join("net").join("feed.key");
            anyhow::ensure!(!path.exists(), "{} exists: never overwritten", path.display());
            std::fs::create_dir_all(path.parent().expect("dir"))?;
            let mut secret = [0u8; 32];
            getrandom::getrandom(&mut secret).map_err(|e| anyhow::anyhow!("{e}"))?;
            let k = FeedKey::from_secret(secret);
            std::fs::write(&path, hex::encode(secret))?;
            println!("{{\"feed_public\":\"{}\"}}", hex::encode(k.public()));
        }
        Some("import") => {
            let (cfg, _) = GuardianConfig::load(&PathBuf::from(get("--config")?))?;
            let pass = passphrase(get("--passphrase-fd")?.parse()?)?;
            let sign = cfg.state_dir.join("sign");
            let machine = open_machine_key(&std::fs::read(sign.join(format!("machine-g{}.key.sealed", cfg.generation)))?, &pass)?;
            let blob = std::fs::read(get("--sealed")?)?;
            let payload = open_sealed(&machine, &blob, &share_aad(cfg.generation, cfg.index))?;
            let scalar: [u8; 32] = payload[8..40].try_into()?;
            let share = Share::from_bytes(cfg.index, &scalar).map_err(|e| anyhow::anyhow!("{e}"))?;
            let commitments = cfg.commitments()?;
            anyhow::ensure!(verify_share(&share, &commitments), "the share fails the Feldman commitments");
            let group = cfg.group_key()?;
            anyhow::ensure!(commitments.first() == Some(&group), "C_0 is not the group key");
            let m = test_message(&group, "import");
            let pk = share_public_key(cfg.index, &commitments).map_err(|e| anyhow::anyhow!("{e}"))?;
            anyhow::ensure!(verify(&partial_sign(&share, &m), &m, &pk), "selftest: the partial on the test message fails pk_i");
            drop(share);
            let h0 = HeaderInfo::from_hex(&get("--h0-header")?)?;
            let domain = cfg.domain()?;
            anyhow::ensure!(h0.height == 0 && h0.id == domain.network, "H0 must be the genesis of the configured network here");
            std::fs::write(sign.join(format!("share-g{}.sealed", cfg.generation)), &blob)?;
            let mut j = Journal::open(&sign)?;
            if j.genesis.is_none() {
                use sha2::Digest;
                let mut concat = Vec::new();
                for c in &commitments {
                    concat.extend_from_slice(c);
                }
                j.init_genesis(
                    &GenesisBody {
                        index: cfg.index,
                        generation: cfg.generation,
                        group_key: hex::encode(group),
                        commitments_digest: hex::encode(sha2::Sha256::digest(&concat)),
                        network: hex::encode(domain.network),
                        l1_chain_id: domain.l1_chain_id,
                        l1_bridge: hex::encode(domain.l1_bridge),
                        h0_height: 0,
                        h0_id: hex::encode(h0.id),
                        h0_work: hex::encode(h0.work),
                        created_unix: jetsam_guardian::now_unix(),
                    },
                    h0.tx_root,
                )?;
            }
            println!("{{\"imported\":{},\"share_key\":\"{}\",\"journal_records\":{}}}", cfg.index, hex::encode(pk), j.head().0);
        }
        _ => anyhow::bail!("jg-keys machine-init | feed-init | import"),
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("jg-keys: {e:#}");
        std::process::exit(1);
    }
}
