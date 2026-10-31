// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-sign`: the share, the journal, and the ONLY signing path (design §2.4,
//! rules SG-11 to SG-16, journal §4).
//!
//! The path, in this order, with no other way to a partial:
//! 1. reception of a statement (416 bytes) with the two verdicts already
//!    received for exactly its hash (SG-14);
//! 2. SG-11 (form), SG-15 (volume), SG-12 (independent recomputation from the
//!    raw headers of the observer's verdict) and SG-13 (linear consistency
//!    with the signed chain);
//! 3. `INTENT` (and the `EXTEND` it needs) written and synchronized (SG-16);
//! 4. the partial computed (the single call of `threshold::partial_sign`);
//! 5. `SIGNED` written;
//! 6. the partial released.
//!
//! A statement already signed is answered with the SAME bytes, read from the
//! journal (a re-emission after a crash is not a new signature).

use std::collections::HashMap;

use crate::chain::{merkle_root, HeaderInfo};
use crate::journal::{ChainEntry, ExtendBody, IntentBody, Journal, RecordType, SignedBody};
use crate::threshold::{self, Share};
use crate::{u256, Domain, GuardianStatement, STATEMENT_BYTES};
use jetsam_chain::consensus::guardian::{anchor_leaf_hash, headers_digest};

/// The floor of `sign_depth` (= `CONSENSUS_FINALITY_DEPTH`).
pub const SIGN_DEPTH_FLOOR: u64 = 8;

/// The signer's configuration (from `guardian.toml`, pinned at start).
#[derive(Clone, Debug)]
pub struct SignerConfig {
    pub domain: Domain,
    pub sign_depth: u64,
    /// `genesis_go = <L_g, id(L_g)>`, written by the operator after the GO G8.
    pub genesis_go: Option<(u64, [u8; 32])>,
    /// `genesisLockedHeight` once the bridge genesis is installed.
    pub genesis_locked_height: Option<u64>,
    /// J and `BRIDGE_DEPOSITS_FROM_HEIGHT` (bounds of a genesis, IG-2).
    pub v1_7_height: u64,
    pub deposits_from_height: u64,
    /// `now + not_after_min ≤ notAfter ≤ now + not_after_max` (SG-11).
    pub not_after_min: u64,
    pub not_after_max: u64,
    /// Age of a verdict at most (SG-14).
    pub verdict_max_age: u64,
    /// SG-15.
    pub max_per_height: usize,
    pub max_per_hour: usize,
}

impl SignerConfig {
    /// The design's defaults for `domain`.
    pub fn defaults(domain: Domain, v1_7_height: u64, deposits_from_height: u64) -> Self {
        Self {
            domain,
            sign_depth: 8,
            genesis_go: None,
            genesis_locked_height: None,
            v1_7_height,
            deposits_from_height,
            not_after_min: 300,
            not_after_max: 3_480,
            verdict_max_age: 60,
            max_per_height: 4,
            max_per_hour: 240,
        }
    }
}

/// A verdict of `jg-observe` ("O", with the raw headers `T_max + 1 ..=
/// s.height`) or of `jg-net` ("W").
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Verdict {
    /// keccak of the 416 bytes.
    #[serde(with = "crate::checkpoint::hex32")]
    pub statement_hash: [u8; 32],
    pub ok: bool,
    pub code: String,
    pub at_unix: u64,
    #[serde(default)]
    pub headers: Vec<String>,
}

/// Why the signer refuses (codes of the design, in English).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// SG-11.
    BadForm(String),
    GenesisNotAuthorized,
    /// SG-14.
    MissingVerdict,
    /// SG-15.
    RateLimited,
    /// SG-12.
    RecomputeMismatch(String),
    /// SG-13 (a refusal).
    JournalConflict,
    /// SG-13: the statement contradicts the journal while the node and the
    /// sources confirm it: the node changed history beyond finality. HALT.
    HaltDeepDivergence,
    /// SG-16.
    JournalIo(String),
    /// Below the trust anchor H0.
    BelowTrustAnchor,
    /// The signer is halted (persisted).
    Halted(String),
    /// TEST ONLY: a simulated crash between INTENT and the partial.
    Crashed,
    /// REPRISE (design §4.4): the journal is behind or off what was published.
    Reprise(String),
}

impl Refusal {
    pub fn code(&self) -> String {
        match self {
            Self::BadForm(why) => format!("BAD_FORM:{why}"),
            Self::GenesisNotAuthorized => "GENESIS_NOT_AUTHORIZED".into(),
            Self::MissingVerdict => "MISSING_VERDICT".into(),
            Self::RateLimited => "RATE_LIMITED".into(),
            Self::RecomputeMismatch(why) => format!("RECOMPUTE_MISMATCH:{why}"),
            Self::JournalConflict => "JOURNAL_CONFLICT".into(),
            Self::HaltDeepDivergence => "HALT_DEEP_DIVERGENCE".into(),
            Self::JournalIo(why) => format!("JOURNAL_IO:{why}"),
            Self::BelowTrustAnchor => "BELOW_TRUST_ANCHOR".into(),
            Self::Halted(cause) => format!("HALTED:{cause}"),
            Self::Crashed => "CRASHED".into(),
            Self::Reprise(why) => format!("REPRISE:{why}"),
        }
    }
}

/// The three forms of a statement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StatementKind {
    /// `anchorRoot = depositRoot = depositCount = notAfter = 0` (§2.5).
    AttestationOnly,
    /// Empty roots, count 0: the bridge genesis (§2.5, A1 B1).
    Genesis,
    /// A checkpoint co-signature (describes the locked block L).
    Cosign,
}

/// The form of a statement (no other check).
pub fn kind_of(s: &GuardianStatement) -> StatementKind {
    let empty = crate::chain::empty_root();
    if s.anchor_root == [0u8; 32] && s.deposit_root == [0u8; 32] && s.deposit_count == 0 && s.not_after == 0 {
        StatementKind::AttestationOnly
    } else if s.anchor_root == empty && s.deposit_root == empty && s.deposit_count == 0 {
        StatementKind::Genesis
    } else {
        StatementKind::Cosign
    }
}

/// TEST ONLY fault injection.
#[derive(Clone, Copy, Debug, Default)]
pub struct Faults {
    /// Stop between the INTENT and the partial, as a `kill -9` would.
    pub crash_after_intent: bool,
}

/// The signer.
pub struct Signer {
    cfg: SignerConfig,
    journal: Journal,
    share: Share,
    /// The Feldman commitments the share was checked against at start.
    pub commitments: Vec<[u8; 48]>,
    verdicts_o: HashMap<[u8; 32], Verdict>,
    verdicts_w: HashMap<[u8; 32], Verdict>,
    reprise: Option<String>,
    pub faults: Faults,
}

impl Signer {
    /// Start: the share must match the commitments, `sign_depth ≥ 8`, the
    /// journal must have its GENESIS record.
    pub fn start(cfg: SignerConfig, journal: Journal, share: Share, commitments: Vec<[u8; 48]>) -> anyhow::Result<Self> {
        anyhow::ensure!(
            cfg.sign_depth >= SIGN_DEPTH_FLOOR,
            "sign_depth {} is below the floor {SIGN_DEPTH_FLOOR}: the signer does not start",
            cfg.sign_depth
        );
        anyhow::ensure!(
            threshold::verify_share(&share, &commitments),
            "share {} does not match the Feldman commitments: the signer does not start",
            share.index
        );
        anyhow::ensure!(journal.genesis.is_some() && journal.t_max().is_some(), "the journal has no GENESIS record");
        Ok(Self { cfg, journal, share, commitments, verdicts_o: HashMap::new(), verdicts_w: HashMap::new(), reprise: None, faults: Faults::default() })
    }

    /// A verdict of `jg-observe`.
    pub fn put_observer_verdict(&mut self, v: Verdict) {
        self.verdicts_o.insert(v.statement_hash, v);
    }

    /// A verdict of `jg-net`.
    pub fn put_witness_verdict(&mut self, v: Verdict) {
        self.verdicts_w.insert(v.statement_hash, v);
    }

    /// `(T_max, id(T_max))`.
    pub fn t_max(&self) -> (u64, [u8; 32]) {
        let t = self.journal.t_max().expect("a started signer has its anchor");
        (t, self.journal.entry(t).expect("t_max").id)
    }

    /// The halt cause, if halted.
    pub fn halted(&self) -> Option<&str> {
        self.journal.halted.as_deref()
    }

    /// The journal (read).
    pub fn journal(&self) -> &Journal {
        &self.journal
    }

    /// The share index.
    pub fn index(&self) -> u32 {
        self.share.index
    }

    /// THE signing path (see the module comment).
    pub fn sign(&mut self, statement: &[u8; STATEMENT_BYTES], now: u64) -> Result<[u8; 96], Refusal> {
        if let Some(cause) = &self.journal.halted {
            return Err(Refusal::Halted(cause.clone()));
        }
        if let Some(why) = &self.reprise {
            return Err(Refusal::Reprise(why.clone()));
        }
        // SG-11: form.
        let s = GuardianStatement::decode(statement).map_err(|_| Refusal::BadForm("non-canonical".into()))?;
        if !self.cfg.domain.matches(&s) {
            return Err(Refusal::BadForm("foreign domain, network, L1 or schema".into()));
        }
        let kind = kind_of(&s);
        self.check_form(&s, kind, now)?;
        let hash = s.message();
        // SG-14: both verdicts, for exactly this statement, fresh.
        let fresh = |v: Option<&Verdict>| v.is_some_and(|v| v.ok && v.statement_hash == hash && now <= v.at_unix + self.cfg.verdict_max_age && v.at_unix <= now + 5);
        if !fresh(self.verdicts_o.get(&hash)) || !fresh(self.verdicts_w.get(&hash)) {
            return Err(Refusal::MissingVerdict);
        }
        // A statement already signed: the same bytes again.
        if let Some(p) = self.journal.signed.get(&hash) {
            return Ok(*p);
        }
        // Trust anchor: the digest reads h − 7 .. h.
        if s.height < self.journal.h0 + 7 {
            return Err(Refusal::BelowTrustAnchor);
        }
        // SG-15: volume.
        let intended_here = self.journal.per_height.get(&s.height);
        let known = intended_here.is_some_and(|set| set.contains(&hash));
        if !known && intended_here.is_some_and(|set| set.len() >= self.cfg.max_per_height) {
            return Err(Refusal::RateLimited);
        }
        let last_hour = self.journal.signed_times.iter().filter(|t| **t + 3_600 > now).count();
        if !known && last_hour >= self.cfg.max_per_hour {
            return Err(Refusal::RateLimited);
        }
        // SG-12 / SG-13: recomputation and linear consistency.
        let headers = self.verdicts_o.get(&hash).map(|v| v.headers.clone()).unwrap_or_default();
        let extension = match self.consistency(&s, kind, &headers) {
            Ok(ext) => ext,
            Err(Refusal::JournalConflict) => {
                // the node and the sources confirm what the journal contradicts
                let detail = format!("statement {} at height {}", hex::encode(hash), s.height);
                let _ = self.journal.halt("HALT_DEEP_DIVERGENCE", &detail, now);
                return Err(Refusal::HaltDeepDivergence);
            }
            Err(other) => return Err(other),
        };
        // SG-16: INTENT (with its EXTEND) durable before the partial exists.
        let mut records = Vec::new();
        if !extension.is_empty() {
            let from = self.journal.t_max().expect("anchor") + 1;
            let entries = extension.iter().map(|e| (hex::encode(e.id), hex::encode(e.tx_root), hex::encode(e.work))).collect();
            records.push((RecordType::Extend, serde_json::to_vec(&ExtendBody { from, entries }).expect("json")));
        }
        let intent = IntentBody {
            statement: hex::encode(statement),
            statement_hash: hex::encode(hash),
            verdict_o: self.verdicts_o.get(&hash).map(|v| v.code.clone()).unwrap_or_default(),
            verdict_w: self.verdicts_w.get(&hash).map(|v| v.code.clone()).unwrap_or_default(),
            at_unix: now,
        };
        records.push((RecordType::Intent, serde_json::to_vec(&intent).expect("json")));
        self.journal.append_all(&records).map_err(|e| Refusal::JournalIo(e.to_string()))?;
        if self.faults.crash_after_intent {
            return Err(Refusal::Crashed);
        }
        // The single signing call.
        let partial = threshold::partial_sign(&self.share, &hash);
        let signed = SignedBody { statement_hash: hex::encode(hash), partial: hex::encode(partial), at_unix: now };
        self.journal
            .append(RecordType::Signed, serde_json::to_vec(&signed).expect("json"))
            .map_err(|e| Refusal::JournalIo(e.to_string()))?;
        Ok(partial)
    }

    /// SG-11 beyond the domain words.
    fn check_form(&self, s: &GuardianStatement, kind: StatementKind, now: u64) -> Result<(), Refusal> {
        let window = |n: u64| now + self.cfg.not_after_min <= n && n <= now + self.cfg.not_after_max;
        match kind {
            StatementKind::AttestationOnly => Ok(()),
            StatementKind::Genesis => {
                let authorized = self.cfg.genesis_go == Some((s.height, s.block_id))
                    && self.cfg.v1_7_height <= s.height
                    && s.height <= self.cfg.deposits_from_height;
                if !authorized {
                    return Err(Refusal::GenesisNotAuthorized);
                }
                if !window(s.not_after) {
                    return Err(Refusal::BadForm("notAfter outside the window".into()));
                }
                Ok(())
            }
            StatementKind::Cosign => {
                if s.not_after == 0 || !window(s.not_after) {
                    return Err(Refusal::BadForm("notAfter outside the window".into()));
                }
                if s.anchor_root == [0u8; 32] {
                    return Err(Refusal::BadForm("a co-signature has an anchor root".into()));
                }
                match self.cfg.genesis_locked_height {
                    Some(gl) if s.height > gl => Ok(()),
                    _ => Err(Refusal::BadForm("no bridge genesis below this height".into())),
                }
            }
        }
    }

    /// SG-12 (recompute from the raw headers) and SG-13 (the signed chain):
    /// the new entries above `T_max`, or why not.
    fn consistency(&self, s: &GuardianStatement, kind: StatementKind, headers: &[String]) -> Result<Vec<ChainEntry>, Refusal> {
        let t_max = self.journal.t_max().expect("anchor");
        let mut ext: Vec<ChainEntry> = Vec::new();
        if s.height > t_max {
            let mut prev = self.journal.entry(t_max).expect("t_max").id;
            let mut work = self.journal.entry(t_max).expect("t_max").work;
            let parsed: Vec<HeaderInfo> = headers
                .iter()
                .map(|h| HeaderInfo::from_hex(h))
                .collect::<Result<_, _>>()
                .map_err(|e| Refusal::RecomputeMismatch(format!("header: {e}")))?;
            if parsed.len() as u64 != s.height - t_max {
                return Err(Refusal::RecomputeMismatch("headers do not cover T_max + 1 ..= height".into()));
            }
            for (i, h) in parsed.iter().enumerate() {
                if h.height != t_max + 1 + i as u64 || h.prev != prev {
                    return Err(Refusal::RecomputeMismatch("headers do not link to the signed tip".into()));
                }
                work = u256::add(&work, &h.work);
                ext.push(ChainEntry { id: h.id, tx_root: h.tx_root, work });
                prev = h.id;
            }
        }
        let entry = |x: u64| -> Option<ChainEntry> {
            if x <= t_max {
                self.journal.entry(x)
            } else {
                ext.get((x - t_max - 1) as usize).copied()
            }
        };
        // What the chain (signed, then extended) says about s.height.
        let at = entry(s.height).expect("covered");
        let mut ids = [[0u8; 32]; 8];
        for (i, slot) in ids.iter_mut().enumerate() {
            *slot = entry(s.height - 7 + i as u64).ok_or(Refusal::BelowTrustAnchor)?.id;
        }
        let anchor = match kind {
            StatementKind::Cosign => {
                let gl = self.cfg.genesis_locked_height.expect("checked by the form");
                let leaves: Option<Vec<[u8; 32]>> = (gl + 1..=s.height)
                    .map(|x| entry(x).map(|e| anchor_leaf_hash(x, &e.id, &e.tx_root)))
                    .collect();
                Some(merkle_root(&leaves.ok_or(Refusal::BelowTrustAnchor)?))
            }
            _ => None,
        };
        let mismatch = if at.id != s.block_id {
            Some("blockId")
        } else if u256::to_be(&at.work) != s.cum_work {
            Some("cumWork")
        } else if headers_digest(&ids) != s.headers_digest {
            Some("headersDigest")
        } else if anchor.is_some_and(|a| a != s.anchor_root) {
            Some("anchorRoot")
        } else {
            None
        };
        match mismatch {
            None => Ok(ext),
            // at or below T_max the signed chain decides: a contradiction
            Some(_) if s.height <= t_max => Err(Refusal::JournalConflict),
            Some(field) => Err(Refusal::RecomputeMismatch(field.into())),
        }
    }

    /// Enter the REPRISE state: every signature refused until restart.
    pub fn enter_reprise(&mut self, why: &str) {
        self.reprise = Some(why.to_string());
    }

    /// The REPRISE reason, if any.
    pub fn reprise(&self) -> Option<&str> {
        self.reprise.as_deref()
    }

    /// A HALT requested by `jg-net` (a second instance of this index) or the operator.
    pub fn halt_external(&mut self, cause: &str, now: u64) {
        let _ = self.journal.halt(cause, "requested over the signer's socket", now);
    }

    /// Wipe the share (SIGTERM, exit): the scalar buffer is zeroized on drop.
    pub fn shutdown(self) {
        drop(self);
    }
}

#[cfg(test)]
#[path = "signer_tests.rs"]
mod tests;
