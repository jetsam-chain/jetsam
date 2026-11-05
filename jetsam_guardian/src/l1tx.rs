// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Signed L1 transactions for the bridge robots (BRIDGE-REST): the production relayer `jg-relay` and the
//! advance robot `jg-fill` hold their OWN hot key and sign their transactions here (EIP-1559, type 2,
//! secp256k1 ECDSA with RFC 6979 nonces, low s), then send the raw bytes (`eth_sendRawTransaction`). No
//! account is unlocked on the endpoint, no external signer is reached.
//!
//! * [`HotKey`]: read from a file that only its owner can read (mode 0600 or 0400, owned by this
//!   process's user), 32 bytes in hex; never printed (its `Debug` shows the address only).
//! * [`Tx1559`]: the transaction, its RLP, its signing hash, the signed bytes (checked byte for byte
//!   against foundry's `cast mktx`, `tests/fixtures/l1tx_vectors.json`).
//! * [`FeePolicy`]: the gas limit is the endpoint's estimate plus a margin, rounded up, never above a cap;
//!   the fees follow the base fee (`2 × base + tip`) within a cap; a replacement raises both by 1/8.
//! * [`Sender`]: nonce, write-ahead of every signed transaction before it is sent (`pending.json`, synced),
//!   receipt, replacement of a transaction left unmined, and the recovery after a crash: a transaction
//!   signed before a stop is found again (mined, or sent again as the same bytes), never signed anew.

use std::path::{Path, PathBuf};
use std::time::Duration;

use k256::ecdsa::{RecoveryId, Signature, SigningKey, VerifyingKey};
use serde_json::{json, Value};

use crate::l1::{hex_u64, Receipt, L1};

/// A secp256k1 key held by a robot.
pub struct HotKey {
    key: SigningKey,
    address: [u8; 20],
}

impl std::fmt::Debug for HotKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HotKey(0x{})", hex::encode(self.address))
    }
}

/// The L1 address of a public key: the last 20 bytes of keccak256 of its 64 uncompressed bytes.
fn address_of(vk: &VerifyingKey) -> [u8; 20] {
    let point = vk.to_encoded_point(false);
    crate::keccak256(&point.as_bytes()[1..])[12..].try_into().expect("20 bytes")
}

impl HotKey {
    /// From the 32 secret bytes (refused: 0 and anything ≥ the group order).
    pub fn from_bytes(secret: &[u8; 32]) -> anyhow::Result<Self> {
        let key = SigningKey::from_bytes(secret.into()).map_err(|_| anyhow::anyhow!("not a secp256k1 secret key"))?;
        let address = address_of(key.verifying_key());
        Ok(Self { key, address })
    }

    /// From a file of 64 hex characters (an optional `0x`, surrounding white space allowed) that only its
    /// owner, this process's user, can read: mode 0600 or 0400.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        use std::os::unix::fs::MetadataExt;
        let meta = std::fs::metadata(path).map_err(|e| anyhow::anyhow!("hot key {}: {e}", path.display()))?;
        anyhow::ensure!(
            meta.mode() & 0o077 == 0,
            "hot key {}: mode {:o}, it must be readable by its owner only (0600 or 0400)",
            path.display(),
            meta.mode() & 0o777
        );
        // SAFETY: geteuid has no precondition.
        let euid = unsafe { libc::geteuid() };
        anyhow::ensure!(meta.uid() == euid, "hot key {}: owned by uid {}, not by this process's uid {euid}", path.display(), meta.uid());
        let text = zeroize::Zeroizing::new(std::fs::read_to_string(path)?);
        let body = text.trim().trim_start_matches("0x");
        anyhow::ensure!(body.len() == 64 && body.bytes().all(|b| b.is_ascii_hexdigit()), "hot key {}: not 32 bytes of hex", path.display());
        let mut secret = zeroize::Zeroizing::new([0u8; 32]);
        hex::decode_to_slice(body, secret.as_mut()).map_err(|_| anyhow::anyhow!("hot key {}: not hex", path.display()))?;
        Self::from_bytes(&secret).map_err(|e| anyhow::anyhow!("hot key {}: {e}", path.display()))
    }

    pub fn address(&self) -> [u8; 20] {
        self.address
    }
}

/// Minimal RLP (the Ethereum yellow paper, appendix B): byte strings, unsigned integers, lists.
pub mod rlp {
    fn length_prefix(len: usize, short: u8, long: u8) -> Vec<u8> {
        if len < 56 {
            return vec![short + len as u8];
        }
        let be = (len as u64).to_be_bytes();
        let skip = be.iter().take_while(|b| **b == 0).count();
        let mut out = vec![long + (8 - skip) as u8];
        out.extend_from_slice(&be[skip..]);
        out
    }

    /// A byte string.
    pub fn bytes(b: &[u8]) -> Vec<u8> {
        if b.len() == 1 && b[0] < 0x80 {
            return b.to_vec();
        }
        let mut out = length_prefix(b.len(), 0x80, 0xb7);
        out.extend_from_slice(b);
        out
    }

    /// An unsigned integer: its big-endian bytes without leading zeros (0 is the empty string).
    pub fn uint(x: u128) -> Vec<u8> {
        let be = x.to_be_bytes();
        let skip = be.iter().take_while(|b| **b == 0).count();
        bytes(&be[skip..])
    }

    /// A list of already encoded items.
    pub fn list(items: &[Vec<u8>]) -> Vec<u8> {
        let len: usize = items.iter().map(Vec::len).sum();
        let mut out = length_prefix(len, 0xc0, 0xf7);
        for i in items {
            out.extend_from_slice(i);
        }
        out
    }
}

/// An EIP-1559 transaction of a robot: no value (the bridge's calls carry no ether), no access list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tx1559 {
    pub chain_id: u64,
    pub nonce: u64,
    pub max_priority_fee_per_gas: u128,
    pub max_fee_per_gas: u128,
    pub gas_limit: u64,
    pub to: [u8; 20],
    pub data: Vec<u8>,
}

/// The bytes to send, their hash (the transaction hash) and the signature `(y parity, r, s)`.
#[derive(Clone, Debug)]
pub struct SignedTx {
    pub raw: Vec<u8>,
    pub hash: [u8; 32],
    pub signature: (u8, [u8; 32], [u8; 32]),
}

/// A 32-byte big-endian scalar as an RLP integer (no leading zeros).
fn scalar(x: &[u8; 32]) -> Vec<u8> {
    let skip = x.iter().take_while(|b| **b == 0).count();
    rlp::bytes(&x[skip..])
}

impl Tx1559 {
    fn fields(&self) -> Vec<Vec<u8>> {
        vec![
            rlp::uint(u128::from(self.chain_id)),
            rlp::uint(u128::from(self.nonce)),
            rlp::uint(self.max_priority_fee_per_gas),
            rlp::uint(self.max_fee_per_gas),
            rlp::uint(u128::from(self.gas_limit)),
            rlp::bytes(&self.to),
            rlp::uint(0),
            rlp::bytes(&self.data),
            rlp::list(&[]),
        ]
    }

    /// `keccak256(0x02 ‖ rlp([chainId, nonce, maxPriorityFeePerGas, maxFeePerGas, gas, to, 0, data, []]))`.
    pub fn signing_hash(&self) -> [u8; 32] {
        let mut pre = vec![0x02];
        pre.extend_from_slice(&rlp::list(&self.fields()));
        crate::keccak256(&pre)
    }

    /// Signed by `key` (RFC 6979, low s): `0x02 ‖ rlp([..., yParity, r, s])`.
    pub fn sign(&self, key: &HotKey) -> SignedTx {
        let hash = self.signing_hash();
        let (sig, rec): (Signature, RecoveryId) = key.key.sign_prehash_recoverable(&hash).expect("a 32-byte prehash signs");
        // k256 already signs with a low s; if it ever did not, the negated s flips the parity
        let (sig, rec) = match sig.normalize_s() {
            Some(low) => (low, RecoveryId::new(!rec.is_y_odd(), rec.is_x_reduced())),
            None => (sig, rec),
        };
        let r: [u8; 32] = sig.r().to_bytes().into();
        let s: [u8; 32] = sig.s().to_bytes().into();
        let parity = u8::from(rec.is_y_odd());
        let mut fields = self.fields();
        fields.push(rlp::uint(u128::from(parity)));
        fields.push(scalar(&r));
        fields.push(scalar(&s));
        let mut raw = vec![0x02];
        raw.extend_from_slice(&rlp::list(&fields));
        let tx_hash = crate::keccak256(&raw);
        SignedTx { raw, hash: tx_hash, signature: (parity, r, s) }
    }
}

/// The address that signed `hash` with `(parity, r, s)`.
pub fn recover(hash: &[u8; 32], parity: u8, r: &[u8; 32], s: &[u8; 32]) -> anyhow::Result<[u8; 20]> {
    let mut rs = [0u8; 64];
    rs[..32].copy_from_slice(r);
    rs[32..].copy_from_slice(s);
    let sig = Signature::from_slice(&rs).map_err(|_| anyhow::anyhow!("bad signature"))?;
    let rec = RecoveryId::from_byte(parity).ok_or_else(|| anyhow::anyhow!("bad parity"))?;
    let vk = VerifyingKey::recover_from_prehash(hash, &sig, rec).map_err(|_| anyhow::anyhow!("signature does not recover"))?;
    Ok(address_of(&vk))
}

/// How a robot pays for its transactions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FeePolicy {
    /// Added to the endpoint's gas estimate, in percent (10 to 100).
    pub gas_margin_pct: u64,
    /// No transaction above this gas limit.
    pub max_gas: u64,
    /// No fee per gas above this (wei).
    pub max_fee_per_gas: u128,
    /// The priority fee (wei per gas); `None`: the endpoint's `eth_maxPriorityFeePerGas` (the private rehearsal
    /// only: `SignerConfig::check_fees`).
    pub priority_fee: Option<u128>,
    /// A5-13: no priority fee above this (wei per gas), configured, the endpoint's or a replacement's: a cap of the
    /// tip of its own, far under `max_fee_per_gas` (a lying endpoint made the tip ≈ the whole fee cap).
    pub max_priority_fee_per_gas: u128,
}

impl Default for FeePolicy {
    fn default() -> Self {
        // 25 %: `advanceBatch` and `claimDeposit` vary with fresh storage slots and recipients; the estimate
        // is taken on the pending block, the margin covers the state moving before inclusion. A tip of 10 gwei at
        // most (A5-13).
        Self { gas_margin_pct: 25, max_gas: 3_000_000, max_fee_per_gas: 200_000_000_000, priority_fee: None, max_priority_fee_per_gas: 10_000_000_000 }
    }
}

impl FeePolicy {
    /// The margin within 10..=100 %, a gas cap at least that of a plain transaction, a non-zero fee cap.
    pub fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!((10..=100).contains(&self.gas_margin_pct), "gas_margin_pct {} outside 10..=100", self.gas_margin_pct);
        anyhow::ensure!(self.max_gas >= 21_000, "max_gas {} below 21 000", self.max_gas);
        anyhow::ensure!(self.max_fee_per_gas > 0, "max_fee_per_gas is 0");
        if let Some(p) = self.priority_fee {
            anyhow::ensure!(p <= self.max_priority_fee_per_gas, "priority fee {p} above its cap {}", self.max_priority_fee_per_gas);
        }
        Ok(())
    }

    /// `⌈estimate × (100 + margin) / 100⌉`; above `max_gas` the transaction is refused (with less than
    /// its margin it might run out of gas: it is not sent).
    pub fn gas_limit(&self, estimate: u64) -> anyhow::Result<u64> {
        let want = (u128::from(estimate) * u128::from(100 + self.gas_margin_pct)).div_ceil(100);
        anyhow::ensure!(
            want <= u128::from(self.max_gas),
            "gas estimate {estimate} with its {} % margin ({want}) exceeds max_gas {}: not sent",
            self.gas_margin_pct,
            self.max_gas
        );
        Ok(want as u64)
    }

    /// `(priority fee, max fee)` for `base` (the latest base fee) and the endpoint's `tip`: `2 × base +
    /// priority`, capped; refused when the priority is above its own cap (A5-13) or the cap cannot pay `base +
    /// priority`.
    pub fn fees(&self, base: u128, tip: u128) -> anyhow::Result<(u128, u128)> {
        let prio = self.priority_fee.unwrap_or(tip);
        anyhow::ensure!(prio <= self.max_priority_fee_per_gas, "priority fee {prio} wei/gas above its cap {}: not sent", self.max_priority_fee_per_gas);
        anyhow::ensure!(
            base.saturating_add(prio) <= self.max_fee_per_gas,
            "base fee {base} + priority {prio} wei/gas above the cap {}: not sent",
            self.max_fee_per_gas
        );
        Ok((prio, base.saturating_mul(2).saturating_add(prio).min(self.max_fee_per_gas)))
    }

    /// The fees of a replacement of a transaction sent with `old`: both raised by 1/8, rounded up, within
    /// their caps (the tip's own, A5-13); refused when a cap leaves less than the 10 % raise the endpoints
    /// require.
    pub fn bump(&self, old: (u128, u128)) -> anyhow::Result<(u128, u128)> {
        let up = |x: u128| x.saturating_mul(9).div_ceil(8).max(x + 1);
        let tenth = |x: u128| x.saturating_mul(11).div_ceil(10);
        let max_fee = up(old.1).min(self.max_fee_per_gas);
        let prio = up(old.0).min(max_fee).min(self.max_priority_fee_per_gas);
        anyhow::ensure!(
            max_fee >= tenth(old.1) && prio >= tenth(old.0),
            "fees ({}, {}) cannot be raised by 10 % under the cap {}",
            old.0,
            old.1,
            self.max_fee_per_gas
        );
        Ok((prio, max_fee))
    }
}

/// A signed transaction written before it was sent: what [`Sender`] finds again after a stop.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Pending {
    /// What the transaction does, for the caller (e.g. `advanceBatch`, `claimDeposit`).
    pub label: String,
    pub nonce: u64,
    /// `(hash, raw)` of every version sent with this nonce (a replacement adds one), `0x` hex.
    pub sent: Vec<(String, String)>,
    /// `(priority fee, max fee)` of the last version.
    pub fees: (u128, u128),
    pub to: String,
    pub data: String,
    pub gas_limit: u64,
    /// This host's time of the last version.
    pub sent_at: u64,
}

/// What became of a transaction found in `pending.json`.
#[derive(Debug)]
pub enum Recovered {
    /// Nothing was pending.
    Nothing,
    /// It was mined (its receipt, its label).
    Mined(Receipt, String),
    /// Its nonce was used by another transaction (none of its versions): it never will be mined.
    Dropped(String),
    /// M8: found unmined, its label one of [`Sender::cancel_at_recovery`]: its nonce went to a zero-value transfer to
    /// the key itself, it never will be mined.
    Cancelled(String),
}

/// Sends a robot's transactions, signed by its key, one at a time.
#[derive(Debug)]
pub struct Sender {
    pub key: HotKey,
    pub chain_id: u64,
    pub policy: FeePolicy,
    /// `pending.json`: the transaction in flight, at most one.
    pub pending_path: PathBuf,
    /// A version left unmined this long is replaced (higher fees, same nonce).
    pub replace_after: Duration,
    /// No receipt after this long: an error (the transaction stays pending, settled at the next send).
    pub receipt_timeout: Duration,
    /// M8 (re-audit n°2): the labels whose transaction, found unmined at a recovery, is cancelled (its nonce given to a
    /// zero-value transfer to the key itself, fees raised) instead of being sent again: what was decided before the
    /// stop may no longer hold (`jg-fill`'s advances: `fill::CANCEL_AT_RECOVERY`). Empty: every one is sent again.
    pub cancel_at_recovery: Vec<String>,
}

fn hex0x(b: &[u8]) -> String {
    format!("0x{}", hex::encode(b))
}

fn quantity(v: &Value) -> anyhow::Result<u128> {
    let text = v.as_str().ok_or_else(|| anyhow::anyhow!("not a quantity: {v}"))?;
    Ok(u128::from_str_radix(text.trim_start_matches("0x"), 16)?)
}

impl Sender {
    /// `pending.json` under `state_dir`.
    pub fn new(key: HotKey, chain_id: u64, policy: FeePolicy, state_dir: &Path) -> anyhow::Result<Self> {
        policy.check()?;
        Ok(Self {
            key,
            chain_id,
            policy,
            pending_path: state_dir.join("pending.json"),
            replace_after: Duration::from_secs(180),
            receipt_timeout: Duration::from_secs(1_800),
            cancel_at_recovery: Vec::new(),
        })
    }

    pub fn address(&self) -> [u8; 20] {
        self.key.address()
    }

    /// The transaction in flight, if any.
    pub fn pending(&self) -> anyhow::Result<Option<Pending>> {
        match std::fs::read_to_string(&self.pending_path) {
            Ok(text) => Ok(Some(serde_json::from_str(&text)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Written and synced (file and directory) before the transaction leaves this host.
    fn write_pending(&self, p: &Pending) -> anyhow::Result<()> {
        use std::io::Write;
        let tmp = self.pending_path.with_extension("tmp");
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(serde_json::to_string(p)?.as_bytes())?;
        f.sync_all()?;
        std::fs::rename(&tmp, &self.pending_path)?;
        if let Some(dir) = self.pending_path.parent() {
            std::fs::File::open(dir)?.sync_all()?;
        }
        Ok(())
    }

    fn clear_pending(&self) -> anyhow::Result<()> {
        match std::fs::remove_file(&self.pending_path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// `eth_sendRawTransaction`; an endpoint that already holds these bytes is not an error.
    fn broadcast(&self, l1: &L1, raw_hex: &str) -> anyhow::Result<()> {
        match l1.rpc("eth_sendRawTransaction", json!([raw_hex])) {
            Ok(_) => Ok(()),
            Err(e) => {
                let text = format!("{e:#}").to_lowercase();
                if text.contains("already known") || text.contains("already imported") || text.contains("known transaction") {
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    fn receipt_of(&self, l1: &L1, hash: &str) -> anyhow::Result<Option<Receipt>> {
        let r = l1.rpc("eth_getTransactionReceipt", json!([hash]))?;
        if r.is_null() {
            return Ok(None);
        }
        let block_number = hex_u64(&r["blockNumber"])?;
        let b = l1.rpc("eth_getBlockByNumber", json!([format!("0x{block_number:x}"), false]))?;
        Ok(Some(Receipt {
            tx_hash: hash.to_string(),
            status: hex_u64(&r["status"])? == 1,
            gas_used: hex_u64(&r["gasUsed"])?,
            block_number,
            block_timestamp: hex_u64(&b["timestamp"])?,
            logs: r["logs"].as_array().cloned().unwrap_or_default(),
        }))
    }

    fn mined_nonce(&self, l1: &L1) -> anyhow::Result<u64> {
        hex_u64(&l1.rpc("eth_getTransactionCount", json!([hex0x(&self.address()), "latest"]))?)
    }

    /// At start (and before any new send): the transaction of `pending.json`, if any, is found mined, or
    /// found dropped (its nonce used by another), or sent again as the same bytes and waited for.
    pub fn recover(&self, l1: &L1) -> anyhow::Result<Recovered> {
        let Some(p) = self.pending()? else { return Ok(Recovered::Nothing) };
        for (hash, _) in &p.sent {
            if let Some(r) = self.receipt_of(l1, hash)? {
                self.clear_pending()?;
                return Ok(Recovered::Mined(r, p.label));
            }
        }
        if self.mined_nonce(l1)? > p.nonce {
            self.clear_pending()?;
            return Ok(Recovered::Dropped(p.label));
        }
        // M8: never mined, its nonce still free, and not to be sent again: cancelled (a transfer of 0 to the key itself
        // at its nonce, fees raised so that it replaces the transaction in a pool still holding it), written ahead as a
        // version of its own label ("…, cancelled": a later recovery sends that one again); one of the transaction's
        // own versions may still be mined first
        if self.cancel_at_recovery.contains(&p.label) {
            let mut p = p;
            let label = p.label.clone();
            let own: Vec<String> = p.sent.iter().map(|(h, _)| h.clone()).collect();
            let fees = self.policy.bump(p.fees)?;
            let me = hex0x(&self.address());
            let signed = self.tx(p.nonce, fees, 21_000, &me, "")?.sign(&self.key);
            p.sent.push((hex0x(&signed.hash), hex0x(&signed.raw)));
            p.label = format!("{label}, cancelled");
            p.fees = fees;
            p.to = me;
            p.data = "0x".into();
            p.gas_limit = 21_000;
            p.sent_at = crate::now_unix();
            self.write_pending(&p)?;
            self.broadcast(l1, &hex0x(&signed.raw))?;
            return match self.wait(l1, p)? {
                Some(r) if own.contains(&r.tx_hash) => Ok(Recovered::Mined(r, label)),
                Some(_) => Ok(Recovered::Cancelled(label)),
                None => Ok(Recovered::Dropped(label)),
            };
        }
        // never mined, its nonce still free: the last version again (an endpoint restarted may have lost it)
        if let Some((_, raw)) = p.sent.last() {
            self.broadcast(l1, raw)?;
        }
        let label = p.label.clone();
        match self.wait(l1, p)? {
            Some(r) => Ok(Recovered::Mined(r, label)),
            None => Ok(Recovered::Dropped(label)),
        }
    }

    /// Wait for one of the versions of `p` (`None`: its nonce was used by another transaction), replacing
    /// it with higher fees when it stays unmined.
    fn wait(&self, l1: &L1, mut p: Pending) -> anyhow::Result<Option<Receipt>> {
        let start = std::time::Instant::now();
        let mut since = std::time::Instant::now();
        loop {
            for (hash, _) in p.sent.iter().rev() {
                if let Some(r) = self.receipt_of(l1, hash)? {
                    self.clear_pending()?;
                    return Ok(Some(r));
                }
            }
            if self.mined_nonce(l1)? > p.nonce {
                // mined between the two reads: one more look at the versions before calling it dropped
                for (hash, _) in p.sent.iter().rev() {
                    if let Some(r) = self.receipt_of(l1, hash)? {
                        self.clear_pending()?;
                        return Ok(Some(r));
                    }
                }
                self.clear_pending()?;
                return Ok(None);
            }
            anyhow::ensure!(
                start.elapsed() < self.receipt_timeout,
                "no receipt for {} (nonce {}) after {:?}: still pending",
                p.label,
                p.nonce,
                self.receipt_timeout
            );
            if since.elapsed() >= self.replace_after {
                match self.policy.bump(p.fees) {
                    Ok(fees) => {
                        let signed = self.tx(p.nonce, fees, p.gas_limit, &p.to, &p.data)?.sign(&self.key);
                        p.sent.push((hex0x(&signed.hash), hex0x(&signed.raw)));
                        p.fees = fees;
                        p.sent_at = crate::now_unix();
                        self.write_pending(&p)?;
                        self.broadcast(l1, &hex0x(&signed.raw))?;
                    }
                    Err(e) => eprintln!("{} l1tx: {} (nonce {}) not replaced: {e:#}", crate::now_unix(), p.label, p.nonce),
                }
                since = std::time::Instant::now();
            }
            std::thread::sleep(Duration::from_millis(250));
        }
    }

    fn tx(&self, nonce: u64, fees: (u128, u128), gas_limit: u64, to: &str, data: &str) -> anyhow::Result<Tx1559> {
        Ok(Tx1559 {
            chain_id: self.chain_id,
            nonce,
            max_priority_fee_per_gas: fees.0,
            max_fee_per_gas: fees.1,
            gas_limit,
            to: hex::decode(to.trim_start_matches("0x"))?.try_into().map_err(|_| anyhow::anyhow!("to: 20 bytes"))?,
            data: hex::decode(data.trim_start_matches("0x"))?,
        })
    }

    /// Sign and send a call of `data` to `to`, and wait for its receipt (a reverted transaction is a
    /// receipt with `status = false`). Refused before signing: a call that reverts on the pending block (its
    /// reason is the error), a gas estimate above the cap, fees above the cap, too little ether for the
    /// gas. A transaction still pending from before (a stop, a timeout) is settled first.
    pub fn send(&self, l1: &L1, to: &str, data: &[u8], label: &str) -> anyhow::Result<Receipt> {
        match self.recover(l1)? {
            Recovered::Nothing => {}
            Recovered::Mined(r, old) => {
                eprintln!("{} l1tx: the pending {old} was mined in block {} (status {})", crate::now_unix(), r.block_number, r.status)
            }
            Recovered::Dropped(old) => eprintln!("{} l1tx: the pending {old} was dropped (its nonce was used)", crate::now_unix()),
            Recovered::Cancelled(old) => eprintln!("{} l1tx: the pending {old} was cancelled, never sent again (M8)", crate::now_unix()),
        }
        let from = hex0x(&self.address());
        let call = json!({"from": from, "to": to, "data": hex0x(data)});
        if let Err(e) = l1.rpc("eth_call", json!([call, "pending"])) {
            anyhow::bail!("reverts: {e}");
        }
        let estimate = hex_u64(&l1.rpc("eth_estimateGas", json!([call, "pending"]))?)?;
        let gas_limit = self.policy.gas_limit(estimate)?;
        let latest = l1.rpc("eth_getBlockByNumber", json!(["latest", false]))?;
        let base = quantity(&latest["baseFeePerGas"]).unwrap_or(0);
        let tip = match self.policy.priority_fee {
            Some(p) => p,
            None => quantity(&l1.rpc("eth_maxPriorityFeePerGas", json!([]))?)?,
        };
        let fees = self.policy.fees(base, tip)?;
        let balance = quantity(&l1.rpc("eth_getBalance", json!([from, "pending"]))?)?;
        let cost = u128::from(gas_limit).saturating_mul(fees.1);
        anyhow::ensure!(balance >= cost, "{from} holds {balance} wei, {label} may cost {cost} (gas {gas_limit} × {} wei): not sent", fees.1);
        let nonce = hex_u64(&l1.rpc("eth_getTransactionCount", json!([from, "pending"]))?)?;
        let signed = self.tx(nonce, fees, gas_limit, to, &hex::encode(data))?.sign(&self.key);
        let p = Pending {
            label: label.to_string(),
            nonce,
            sent: vec![(hex0x(&signed.hash), hex0x(&signed.raw))],
            fees,
            to: to.to_string(),
            data: hex0x(data),
            gas_limit,
            sent_at: crate::now_unix(),
        };
        // write-ahead: a stop after this line finds the transaction again, never signs a second one
        self.write_pending(&p)?;
        self.broadcast(l1, &hex0x(&signed.raw))?;
        self.wait(l1, p)?.ok_or_else(|| anyhow::anyhow!("{label}: nonce {nonce} was used by another transaction, dropped"))
    }
}

/// `[signer]` of `jg-relay` and `jg-fill`.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignerConfig {
    /// `"local"`: this program signs with `key_file` (the only mode of the public networks). `"endpoint"`:
    /// `eth_sendTransaction` from an account the L1 endpoint holds unlocked (the private rehearsal's anvil
    /// only). Absent: `"local"`, except on the private rehearsal where it stays `"endpoint"` (as before).
    pub mode: Option<String>,
    /// The hot key (`HotKey::load`: 0600 or 0400, owned by this user).
    pub key_file: Option<PathBuf>,
    #[serde(default = "default_margin")]
    pub gas_margin_pct: u64,
    #[serde(default = "default_max_gas")]
    pub max_gas: u64,
    /// The cap of the fee per gas, wei.
    #[serde(default = "default_max_fee")]
    pub max_fee_per_gas_wei: u64,
    /// The priority fee, wei per gas. A5-13: required off the private rehearsal (the L1 endpoint's
    /// `eth_maxPriorityFeePerGas` is never taken there); absent on the rehearsal: the endpoint's.
    pub priority_fee_wei: Option<u64>,
    /// A5-13: the cap of the priority fee, wei per gas (10 gwei by default), replacements included; a configured
    /// `priority_fee_wei` above it does not start.
    #[serde(default = "default_max_priority_fee")]
    pub max_priority_fee_wei: u64,
}

fn default_margin() -> u64 {
    FeePolicy::default().gas_margin_pct
}
fn default_max_gas() -> u64 {
    FeePolicy::default().max_gas
}
fn default_max_fee() -> u64 {
    FeePolicy::default().max_fee_per_gas as u64
}
fn default_max_priority_fee() -> u64 {
    FeePolicy::default().max_priority_fee_per_gas as u64
}

/// How a robot's transactions are signed.
#[derive(Debug)]
pub enum L1Tx {
    /// By this program (`Sender`).
    Local(Sender),
    /// By the endpoint, from its unlocked TEST account (the private rehearsal only).
    Endpoint { from: [u8; 20] },
}

impl SignerConfig {
    fn policy(&self) -> FeePolicy {
        FeePolicy {
            gas_margin_pct: self.gas_margin_pct,
            max_gas: self.max_gas,
            max_fee_per_gas: u128::from(self.max_fee_per_gas_wei),
            priority_fee: self.priority_fee_wei.map(u128::from),
            max_priority_fee_per_gas: u128::from(self.max_priority_fee_wei),
        }
    }

    /// A5-13: off the private rehearsal the priority fee is configured, and never above its cap: the rule of
    /// `jg-fill` and of `jg-relay` alike (`open` applies it to every local signer). An error: no start.
    pub fn check_fees(&self, profile: crate::Profile) -> anyhow::Result<()> {
        if profile.strict() {
            anyhow::ensure!(
                self.priority_fee_wei.is_some(),
                "[signer] priority_fee_wei is required on {profile} (A5-13: the tip is never the L1 endpoint's word there)"
            );
        }
        if let Some(p) = self.priority_fee_wei {
            anyhow::ensure!(p <= self.max_priority_fee_wei, "[signer] priority_fee_wei {p} is above max_priority_fee_wei {}", self.max_priority_fee_wei);
        }
        Ok(())
    }

    /// The signer of a robot of `profile` whose transactions come from `address` (`None`: the key's),
    /// its write-ahead under `state_dir`. An error means the program does not start.
    pub fn open(&self, profile: crate::Profile, chain_id: u64, address: Option<[u8; 20]>, state_dir: &Path) -> anyhow::Result<L1Tx> {
        let rehearsal = profile == crate::Profile::PrivateRehearsal;
        let mode = match self.mode.as_deref() {
            Some(m) => m,
            None if rehearsal && self.key_file.is_none() => "endpoint",
            None => "local",
        };
        match mode {
            "local" => {
                let path = self.key_file.as_ref().ok_or_else(|| anyhow::anyhow!("[signer] key_file is required (mode \"local\")"))?;
                self.check_fees(profile)?;
                let key = HotKey::load(path)?;
                if let Some(a) = address {
                    anyhow::ensure!(key.address() == a, "[signer] key_file is the key of 0x{}, not of the configured address 0x{}", hex::encode(key.address()), hex::encode(a));
                }
                Ok(L1Tx::Local(Sender::new(key, chain_id, self.policy(), state_dir)?))
            }
            "endpoint" => {
                anyhow::ensure!(rehearsal, "[signer] mode = \"endpoint\" is refused on {profile}: an unlocked account on an L1 endpoint is the private rehearsal's only");
                anyhow::ensure!(self.key_file.is_none(), "[signer] key_file with mode \"endpoint\": choose one");
                let from = address.ok_or_else(|| anyhow::anyhow!("[signer] mode \"endpoint\" needs the sending address"))?;
                Ok(L1Tx::Endpoint { from })
            }
            other => anyhow::bail!("[signer] mode = {other:?}: \"local\" or \"endpoint\""),
        }
    }
}

impl L1Tx {
    pub fn address(&self) -> [u8; 20] {
        match self {
            L1Tx::Local(s) => s.address(),
            L1Tx::Endpoint { from } => *from,
        }
    }

    /// M8: the labels cancelled when found unmined at a recovery ([`Sender::cancel_at_recovery`], `Local` only).
    pub fn cancel_at_recovery(&mut self, labels: &[&str]) {
        if let L1Tx::Local(s) = self {
            s.cancel_at_recovery = labels.iter().map(|l| l.to_string()).collect();
        }
    }

    /// Settle what a stop left in flight (`Local` only).
    pub fn recover(&self, l1: &L1) -> anyhow::Result<Recovered> {
        match self {
            L1Tx::Local(s) => s.recover(l1),
            L1Tx::Endpoint { .. } => Ok(Recovered::Nothing),
        }
    }

    /// Send a call and wait for its receipt.
    pub fn send(&self, l1: &L1, to: &str, data: &[u8], label: &str) -> anyhow::Result<Receipt> {
        match self {
            L1Tx::Local(s) => s.send(l1, to, data, label),
            L1Tx::Endpoint { from } => l1.send(&hex0x(from), to, data),
        }
    }
}

#[cfg(test)]
#[path = "l1tx_tests.rs"]
mod tests;
