// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The signed feed of a guardian (design §5.1): one line per event, each
//! signed with the guardian's Ed25519 FEED key (low value, held by `jg-net`;
//! never the share). A line is `{"body": "<json text>", "sig": "<hex>"}`;
//! the signature covers `"jetsam-guardian/feed/v1" ‖ body bytes` exactly.
//! Bodies carry `g<i>` and UTC seconds only: no host name, no address, no
//! time zone.
//!
//! Publication points are directories (the rehearsal) the feed is appended
//! to, `feeds/g<i>/<YYYY-MM-DD>.jsonl`; a guardian writes only there.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ed25519_dalek::{Signer as _, SigningKey, Verifier as _, VerifyingKey};

const FEED_DOMAIN: &[u8] = b"jetsam-guardian/feed/v1";

/// A feed key.
pub struct FeedKey(SigningKey);

impl FeedKey {
    pub fn from_secret(secret: [u8; 32]) -> Self {
        Self(SigningKey::from_bytes(&secret))
    }
    pub fn generate() -> anyhow::Result<Self> {
        let mut s = [0u8; 32];
        getrandom::getrandom(&mut s).map_err(|e| anyhow::anyhow!("getrandom: {e}"))?;
        Ok(Self::from_secret(s))
    }
    pub fn public(&self) -> [u8; 32] {
        self.0.verifying_key().to_bytes()
    }
}

/// One signed line for `body` (it must carry `"g": index`).
pub fn sign_line(key: &FeedKey, body: &serde_json::Value) -> String {
    let text = serde_json::to_string(body).expect("json");
    let mut msg = FEED_DOMAIN.to_vec();
    msg.extend_from_slice(text.as_bytes());
    let sig = key.0.sign(&msg);
    serde_json::json!({"body": text, "sig": hex::encode(sig.to_bytes())}).to_string()
}

/// Check a line against the feed keys `keys` (index → public key): the index
/// is the body's `"g"`.
pub fn verify_line(line: &str, keys: &HashMap<u32, [u8; 32]>) -> anyhow::Result<(u32, serde_json::Value)> {
    let outer: serde_json::Value = serde_json::from_str(line)?;
    let text = outer["body"].as_str().ok_or_else(|| anyhow::anyhow!("feed line without body"))?;
    let sig: [u8; 64] = hex::decode(outer["sig"].as_str().unwrap_or_default())?
        .try_into()
        .map_err(|_| anyhow::anyhow!("feed signature: 64 bytes"))?;
    let body: serde_json::Value = serde_json::from_str(text)?;
    let g = body["g"].as_u64().ok_or_else(|| anyhow::anyhow!("feed body without g"))? as u32;
    let key = keys.get(&g).ok_or_else(|| anyhow::anyhow!("no feed key for g{g}"))?;
    let vk = VerifyingKey::from_bytes(key)?;
    let mut msg = FEED_DOMAIN.to_vec();
    msg.extend_from_slice(text.as_bytes());
    vk.verify(&msg, &ed25519_dalek::Signature::from_bytes(&sig))
        .map_err(|_| anyhow::anyhow!("feed line of g{g}: bad signature"))?;
    Ok((g, body))
}

/// The feed file of `index` on `day` (UTC) under a publication point.
pub fn feed_path(point: &Path, index: u32, day: &str) -> PathBuf {
    point.join("feeds").join(format!("g{index}")).join(format!("{day}.jsonl"))
}

/// Append `line` to the feed of `index` on every point; the number reached.
pub fn publish(points: &[PathBuf], index: u32, day: &str, line: &str) -> usize {
    use std::io::Write;
    let mut reached = 0;
    for point in points {
        if !point.is_dir() {
            continue;
        }
        let path = feed_path(point, index, day);
        let ok = (|| -> std::io::Result<()> {
            std::fs::create_dir_all(path.parent().expect("parent"))?;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
            f.sync_data()
        })();
        if ok.is_ok() {
            reached += 1;
        }
    }
    reached
}

/// The UTC day of a Unix time, `YYYY-MM-DD`.
pub fn utc_day(t: u64) -> String {
    // days since 1970-01-01 to a civil date (H. Hinnant's algorithm)
    let z = (t / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
#[path = "feed_tests.rs"]
mod tests;
