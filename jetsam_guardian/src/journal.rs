// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The signer's anti-contradiction journal (design §4).
//!
//! Append only. One record:
//! `"JGJ1" ‖ len u32 BE ‖ type u8 ‖ body (len bytes) ‖ prev 32 ‖ h 32`,
//! `h = SHA-256(prev ‖ type ‖ body)`, `prev` = `h` of the previous record
//! (32 zero bytes for the first). The file is synchronized (fsync) after each
//! write; the directory too when the file is created. Bodies are JSON.
//!
//! Replay rebuilds the **signed chain** (`(id, tx_root, work)` for every
//! height from the trust anchor H0 to `T_max`, the highest height a signature
//! ever covered), the statements signed or intended, and the HALT state:
//! * a tail that is not a whole record is discarded (nothing was released
//!   after it: a partial leaves only after the fsync of its INTENT);
//! * a whole record whose hash does not chain is a broken journal: HALT.
//!
//! One segment per journal in this version (`journal/seg-0.log`); the design's
//! 64 MiB segments are not written (a year of signatures is a few MB).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MAGIC: &[u8; 4] = b"JGJ1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RecordType {
    Genesis = 1,
    Extend = 2,
    Intent = 3,
    Signed = 4,
    Published = 5,
    Halt = 6,
    Resume = 7,
    Restore = 8,
    Rebase = 9,
    Reshare = 10,
    CeremonyTest = 11,
}

impl RecordType {
    pub fn from_u8(x: u8) -> Option<Self> {
        Some(match x {
            1 => Self::Genesis,
            2 => Self::Extend,
            3 => Self::Intent,
            4 => Self::Signed,
            5 => Self::Published,
            6 => Self::Halt,
            7 => Self::Resume,
            8 => Self::Restore,
            9 => Self::Rebase,
            10 => Self::Reshare,
            11 => Self::CeremonyTest,
            _ => return None,
        })
    }
}

/// `GENESIS`: who this journal belongs to and its trust anchor H0.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenesisBody {
    pub index: u32,
    pub generation: u32,
    pub group_key: String,
    /// SHA-256 of the concatenated commitments.
    pub commitments_digest: String,
    pub network: String,
    pub l1_chain_id: u64,
    pub l1_bridge: String,
    pub h0_height: u64,
    pub h0_id: String,
    /// W(H0), u256 little-endian hex.
    pub h0_work: String,
    pub created_unix: u64,
}

/// One height of the signed chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainEntry {
    pub id: [u8; 32],
    pub tx_root: [u8; 32],
    /// u256 little-endian.
    pub work: [u8; 32],
}

/// `EXTEND`: heights `from ..= from + entries.len() − 1`, `(id, tx_root, work)` hex.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtendBody {
    pub from: u64,
    pub entries: Vec<(String, String, String)>,
}

/// `INTENT`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntentBody {
    pub statement: String,
    pub statement_hash: String,
    pub verdict_o: String,
    pub verdict_w: String,
    pub at_unix: u64,
}

/// `SIGNED`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedBody {
    pub statement_hash: String,
    pub partial: String,
    pub at_unix: u64,
}

/// `HALT` / `RESUME` / `REBASE`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteBody {
    pub cause: String,
    pub detail: String,
    pub at_unix: u64,
}

/// The journal, replayed.
pub struct Journal {
    path: PathBuf,
    file: File,
    records: u64,
    head: [u8; 32],
    pub genesis: Option<GenesisBody>,
    /// Heights from H0 (index 0) to `T_max`.
    pub chain: Vec<ChainEntry>,
    pub h0: u64,
    /// Statements with an INTENT (by keccak of the statement).
    pub intents: HashMap<[u8; 32], [u8; 416]>,
    /// Partials released (by statement hash).
    pub signed: HashMap<[u8; 32], [u8; 96]>,
    /// Times of the SIGNED records.
    pub signed_times: Vec<u64>,
    /// Distinct statements intended per height.
    pub per_height: BTreeMap<u64, HashSet<[u8; 32]>>,
    /// Set by a HALT record (and cleared by none: a RESUME is written by an
    /// operator after an investigation and keeps the record).
    pub halted: Option<String>,
    /// The tail bytes discarded at open (an interrupted write).
    pub discarded_tail: usize,
    /// TEST ONLY fault injection: the next write fails.
    pub fail_next_write: bool,
    /// The held `flock` (released on drop).
    _lock: Option<File>,
    /// The hash of every record, in order.
    hashes: Vec<[u8; 32]>,
}

/// One parsed record.
struct Parsed {
    rtype: RecordType,
    body: Vec<u8>,
    hash: [u8; 32],
}

/// Parse the file: the whole records, the length of the good prefix, and
/// whether the rest is a discardable tail (`Ok`) or a broken journal (`Err`).
fn parse(bytes: &[u8]) -> (Vec<Parsed>, usize, Result<(), String>) {
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut prev = [0u8; 32];
    while at < bytes.len() {
        let rest = &bytes[at..];
        let later_magic = |from: usize| rest.len() > from && rest[from..].windows(4).any(|w| w == MAGIC);
        if rest.len() < 9 {
            // a cut header: a tail only if it is a prefix of a record start
            let ok = MAGIC.starts_with(&rest[..rest.len().min(4)]);
            return (out, at, if ok { Ok(()) } else { Err(format!("BROKEN_JOURNAL: garbage at {at}")) });
        }
        if &rest[..4] != MAGIC {
            return (out, at, Err(format!("BROKEN_JOURNAL: no record magic at {at}")));
        }
        let len = u32::from_be_bytes(rest[4..8].try_into().expect("4")) as usize;
        let total = 9 + len + 64;
        if rest.len() < total {
            // an interrupted write is the LAST thing in the file: a later
            // record start means a damaged length in the middle
            return (out, at, if later_magic(4) { Err(format!("BROKEN_JOURNAL: length overruns at {at}")) } else { Ok(()) });
        }
        let Some(rtype) = RecordType::from_u8(rest[8]) else {
            return (out, at, Err(format!("BROKEN_JOURNAL: unknown type at {at}")));
        };
        let body = rest[9..9 + len].to_vec();
        let p: [u8; 32] = rest[9 + len..9 + len + 32].try_into().expect("32");
        let h: [u8; 32] = rest[9 + len + 32..total].try_into().expect("32");
        let (_, expected) = encode_record(&prev, rtype, &body);
        if p != prev || h != expected {
            return (out, at, Err(format!("BROKEN_JOURNAL: hash chain broken at {at}")));
        }
        prev = h;
        out.push(Parsed { rtype, body, hash: h });
        at += total;
    }
    (out, at, Ok(()))
}

fn h32(text: &str) -> anyhow::Result<[u8; 32]> {
    crate::parse32(text)
}

impl Journal {
    /// Open (create) the journal under `dir`, replaying it.
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir.join("journal"))?;
        let lock = OpenOptions::new().create(true).truncate(false).write(true).open(dir.join("lock"))?;
        use std::os::unix::io::AsRawFd;
        // SAFETY: flock on a descriptor we own.
        let r = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        anyhow::ensure!(r == 0, "the journal under {} is locked by another signer instance", dir.display());
        let mut j = Self::replay(dir, true)?;
        j._lock = Some(lock);
        Ok(j)
    }

    fn replay(dir: &Path, writable: bool) -> anyhow::Result<Self> {
        let path = dir.join("journal").join("seg-0.log");
        let created = !path.exists();
        if created && !writable {
            anyhow::bail!("no journal under {}", dir.display());
        }
        let mut file = OpenOptions::new().create(writable).read(true).append(writable).open(&path)?;
        if created {
            File::open(dir.join("journal"))?.sync_all()?;
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let (records, good, status) = parse(&bytes);
        let mut j = Self {
            path: path.clone(),
            file,
            records: 0,
            head: [0u8; 32],
            genesis: None,
            chain: Vec::new(),
            h0: 0,
            intents: HashMap::new(),
            signed: HashMap::new(),
            signed_times: Vec::new(),
            per_height: BTreeMap::new(),
            halted: None,
            discarded_tail: 0,
            fail_next_write: false,
            _lock: None,
            hashes: Vec::new(),
        };
        for r in &records {
            j.apply(r.rtype, &r.body)?;
            j.head = r.hash;
            j.hashes.push(r.hash);
            j.records += 1;
        }
        match status {
            Ok(()) => {
                if good < bytes.len() {
                    j.discarded_tail = bytes.len() - good;
                    if writable {
                        let f = OpenOptions::new().write(true).open(&path)?;
                        f.set_len(good as u64)?;
                        f.sync_all()?;
                    }
                }
            }
            Err(cause) => j.halted = Some(cause),
        }
        Ok(j)
    }

    /// The effect of one record on the replayed state.
    fn apply(&mut self, rtype: RecordType, body: &[u8]) -> anyhow::Result<()> {
        match rtype {
            RecordType::Genesis => {
                let g: GenesisBody = serde_json::from_slice(body)?;
                self.h0 = g.h0_height;
                self.chain = vec![ChainEntry { id: h32(&g.h0_id)?, tx_root: [0u8; 32], work: h32(&g.h0_work)? }];
                self.genesis = Some(g);
            }
            RecordType::Extend => {
                let e: ExtendBody = serde_json::from_slice(body)?;
                anyhow::ensure!(Some(e.from) == self.t_max().map(|t| t + 1), "EXTEND does not follow T_max");
                for (id, tx, w) in &e.entries {
                    self.chain.push(ChainEntry { id: h32(id)?, tx_root: h32(tx)?, work: h32(w)? });
                }
            }
            RecordType::Intent => {
                let i: IntentBody = serde_json::from_slice(body)?;
                let bytes = hex::decode(&i.statement)?;
                let stmt: [u8; 416] = bytes.try_into().map_err(|_| anyhow::anyhow!("INTENT: 416 bytes"))?;
                let hash = crate::keccak256(&stmt);
                let height = u64::from_be_bytes(stmt[184..192].try_into().expect("8"));
                self.per_height.entry(height).or_default().insert(hash);
                self.intents.insert(hash, stmt);
            }
            RecordType::Signed => {
                let s: SignedBody = serde_json::from_slice(body)?;
                let p: [u8; 96] = hex::decode(&s.partial)?.try_into().map_err(|_| anyhow::anyhow!("SIGNED: 96 bytes"))?;
                self.signed.insert(h32(&s.statement_hash)?, p);
                self.signed_times.push(s.at_unix);
            }
            RecordType::Halt => {
                let n: NoteBody = serde_json::from_slice(body)?;
                self.halted = Some(n.cause);
            }
            RecordType::Published
            | RecordType::Resume
            | RecordType::Restore
            | RecordType::Rebase
            | RecordType::Reshare
            | RecordType::CeremonyTest => {}
        }
        Ok(())
    }

    /// Replay without taking the lock (read only: audits, status, tests).
    pub fn replay_readonly(dir: &Path) -> anyhow::Result<Self> {
        Self::replay(dir, false)
    }

    /// The path of the segment.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `(records, head hash)`.
    pub fn head(&self) -> (u64, [u8; 32]) {
        (self.records, self.head)
    }

    /// `T_max` (H0 before any signature).
    pub fn t_max(&self) -> Option<u64> {
        if self.chain.is_empty() {
            None
        } else {
            Some(self.h0 + self.chain.len() as u64 - 1)
        }
    }

    /// The entry at `height`.
    pub fn entry(&self, height: u64) -> Option<ChainEntry> {
        let i = height.checked_sub(self.h0)?;
        self.chain.get(usize::try_from(i).ok()?).copied()
    }

    /// Write several records and synchronize once (an EXTEND and the INTENT
    /// that needs it are written by the same fsync).
    pub fn append_all(&mut self, records: &[(RecordType, Vec<u8>)]) -> anyhow::Result<()> {
        anyhow::ensure!(self._lock.is_some(), "a read-only replay does not write");
        let mut bytes = Vec::new();
        let mut head = self.head;
        let mut new_hashes = Vec::new();
        for (rtype, body) in records {
            let (rec, h) = encode_record(&head, *rtype, body);
            bytes.extend_from_slice(&rec);
            head = h;
            new_hashes.push(h);
        }
        if std::mem::take(&mut self.fail_next_write) {
            anyhow::bail!("journal write failed (injected)");
        }
        self.file.write_all(&bytes)?;
        self.file.sync_all()?;
        for (rtype, body) in records {
            self.apply(*rtype, body)?;
            self.records += 1;
        }
        self.head = head;
        self.hashes.extend(new_hashes);
        Ok(())
    }

    /// Write one record and synchronize.
    pub fn append(&mut self, rtype: RecordType, body: Vec<u8>) -> anyhow::Result<()> {
        self.append_all(&[(rtype, body)])
    }

    /// Write the GENESIS record (once).
    pub fn init_genesis(&mut self, body: &GenesisBody, h0_tx_root: [u8; 32]) -> anyhow::Result<()> {
        anyhow::ensure!(self.genesis.is_none() && self.records == 0, "the journal already has its GENESIS");
        self.append(RecordType::Genesis, serde_json::to_vec(body)?)?;
        if let Some(e) = self.chain.first_mut() {
            e.tx_root = h0_tx_root;
        }
        Ok(())
    }

    /// The hash of record `n` (1-based).
    pub fn record_hash(&self, n: u64) -> Option<[u8; 32]> {
        let i = usize::try_from(n.checked_sub(1)?).ok()?;
        self.hashes.get(i).copied()
    }

    /// A head this signer published (`n` records, hash `head`) compared to
    /// the journal: `None` if it is a prefix of it, else the reason for the
    /// REPRISE state (design §4.4: a restored or lost journal).
    pub fn check_published_head(&self, n: u64, head: &[u8; 32]) -> Option<String> {
        if n == 0 {
            return None;
        }
        match self.record_hash(n) {
            None => Some("JOURNAL_BEHIND_PUBLISHED_HEAD".into()),
            Some(h) if h != *head => Some("JOURNAL_DIVERGES_FROM_PUBLISHED_HEAD".into()),
            Some(_) => None,
        }
    }

    /// Write a HALT record and enter the HALT state.
    pub fn halt(&mut self, cause: &str, detail: &str, now: u64) -> anyhow::Result<()> {
        let body = NoteBody { cause: cause.into(), detail: detail.into(), at_unix: now };
        let r = self.append(RecordType::Halt, serde_json::to_vec(&body)?);
        // halted in memory even if the disk refused the record
        self.halted = Some(cause.to_string());
        r
    }
}

/// Encode one record after `prev`.
pub fn encode_record(prev: &[u8; 32], rtype: RecordType, body: &[u8]) -> (Vec<u8>, [u8; 32]) {
    let mut hasher = Sha256::new();
    hasher.update(prev);
    hasher.update([rtype as u8]);
    hasher.update(body);
    let h: [u8; 32] = hasher.finalize().into();
    let mut out = Vec::with_capacity(4 + 4 + 1 + body.len() + 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.push(rtype as u8);
    out.extend_from_slice(body);
    out.extend_from_slice(prev);
    out.extend_from_slice(&h);
    (out, h)
}

#[cfg(test)]
#[path = "journal_tests.rs"]
mod tests;
