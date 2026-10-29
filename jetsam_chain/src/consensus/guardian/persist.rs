// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! The guardian state at rest (T1.1 §8, §9): its canonical encoding, the
//! undo of one block by pre-images, and the snapshot journal file. Pure
//! byte formats; the tables that hold them are the chain store's.
//!
//! Encoding v1: integers little-endian, `Option` = byte 0 or 1 then the
//! value, tables preceded by their count (u32) and sorted by key. The
//! decoder refuses anything another encoder could not have produced: keys
//! out of order or repeated, an empty position, a zero balance, pending
//! share or entry, an entry id at or above `next_entry_id`, unsorted
//! pending shares, another version, missing or trailing bytes.

use std::collections::BTreeMap;

use jetsam_poseidon2b::primitives::Address;

use super::state::{GuardianState, PayoutEntry, PayoutEntryKind, StakePosition};
use super::{GuardianError, GuardianObjectKind};

/// The encoding version of [`GuardianState`].
pub const GUARDIAN_STATE_VERSION: u8 = 1;
/// The snapshot journal's magic and version.
pub const GUARDIAN_JOURNAL_MAGIC: [u8; 8] = *b"JTMGJN01";
pub const GUARDIAN_JOURNAL_VERSION: u8 = 1;
/// The largest journal file a node reads (§9).
pub const MAX_GUARDIAN_JOURNAL_FILE_BYTES: usize = 1 << 30;

const BAD: GuardianError = GuardianError::NonCanonicalBody;

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], GuardianError> {
        if self.0.len() < n {
            return Err(BAD);
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8, GuardianError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, GuardianError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("4")))
    }
    fn u64(&mut self) -> Result<u64, GuardianError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("8")))
    }
    fn u128(&mut self) -> Result<u128, GuardianError> {
        Ok(u128::from_le_bytes(self.take(16)?.try_into().expect("16")))
    }
    fn key(&mut self) -> Result<[u8; 32], GuardianError> {
        Ok(self.take(32)?.try_into().expect("32"))
    }
    fn option(&mut self) -> Result<Option<u64>, GuardianError> {
        match self.u8()? {
            0 => Ok(None),
            1 => Ok(Some(self.u64()?)),
            _ => Err(BAD),
        }
    }
    fn finish(self) -> Result<(), GuardianError> {
        if self.0.is_empty() {
            Ok(())
        } else {
            Err(BAD)
        }
    }
}

fn put_option(out: &mut Vec<u8>, value: Option<u64>) {
    match value {
        None => out.push(0),
        Some(v) => {
            out.push(1);
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

fn put_count(out: &mut Vec<u8>, n: usize) {
    out.extend_from_slice(&u32::try_from(n).expect("table fits u32").to_le_bytes());
}

fn kind_byte(kind: PayoutEntryKind) -> u8 {
    match kind {
        PayoutEntryKind::Stake => 0,
        PayoutEntryKind::DelegatorReward => 1,
        PayoutEntryKind::GuardianReward => 2,
    }
}

fn kind_of(byte: u8) -> Result<PayoutEntryKind, GuardianError> {
    match byte {
        0 => Ok(PayoutEntryKind::Stake),
        1 => Ok(PayoutEntryKind::DelegatorReward),
        2 => Ok(PayoutEntryKind::GuardianReward),
        _ => Err(BAD),
    }
}

fn put_position(out: &mut Vec<u8>, position: &StakePosition) {
    out.extend_from_slice(&position.active.to_le_bytes());
    out.extend_from_slice(&position.activating.to_le_bytes());
}

fn read_position(r: &mut Reader<'_>) -> Result<StakePosition, GuardianError> {
    Ok(StakePosition {
        active: r.u64()?,
        activating: r.u64()?,
    })
}

fn put_entry(out: &mut Vec<u8>, entry: &PayoutEntry) {
    out.push(kind_byte(entry.kind));
    out.extend_from_slice(&entry.owner.0);
    out.extend_from_slice(&entry.amount.to_le_bytes());
    out.extend_from_slice(&entry.due.to_le_bytes());
}

fn read_entry(r: &mut Reader<'_>) -> Result<PayoutEntry, GuardianError> {
    Ok(PayoutEntry {
        kind: kind_of(r.u8()?)?,
        owner: Address(r.key()?),
        amount: r.u64()?,
        due: r.u64()?,
    })
}

/// The scalar part of the state, in field order.
fn put_scalars(out: &mut Vec<u8>, s: &GuardianState) {
    out.extend_from_slice(&s.vault_floor.to_le_bytes());
    put_option(out, s.revoked_at);
    put_option(out, s.last_attested_height);
    put_option(out, s.attested_period);
    put_option(out, s.activated_at);
    out.extend_from_slice(&s.fund_current.to_le_bytes());
    out.extend_from_slice(&s.confiscated_current.to_le_bytes());
    out.extend_from_slice(&s.carry.to_le_bytes());
    out.extend_from_slice(&s.total_funded.to_le_bytes());
    out.extend_from_slice(&s.total_paid.to_le_bytes());
}

fn read_scalars(r: &mut Reader<'_>, s: &mut GuardianState) -> Result<(), GuardianError> {
    s.vault_floor = r.u64()?;
    s.revoked_at = r.option()?;
    s.last_attested_height = r.option()?;
    s.attested_period = r.option()?;
    s.activated_at = r.option()?;
    s.fund_current = r.u64()?;
    s.confiscated_current = r.u64()?;
    s.carry = r.u64()?;
    s.total_funded = r.u128()?;
    s.total_paid = r.u128()?;
    Ok(())
}

fn put_pending(out: &mut Vec<u8>, pending: &[(u64, u64)]) {
    put_count(out, pending.len());
    for (amount, from) in pending {
        out.extend_from_slice(&amount.to_le_bytes());
        out.extend_from_slice(&from.to_le_bytes());
    }
}

fn read_pending(r: &mut Reader<'_>) -> Result<Vec<(u64, u64)>, GuardianError> {
    let n = r.u32()?;
    let mut pending: Vec<(u64, u64)> = Vec::new();
    for _ in 0..n {
        let (amount, from) = (r.u64()?, r.u64()?);
        if amount == 0 || pending.last().is_some_and(|(_, last)| *last > from) {
            return Err(BAD);
        }
        pending.push((amount, from));
    }
    Ok(pending)
}

impl GuardianState {
    /// The canonical bytes (v1).
    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![GUARDIAN_STATE_VERSION];
        put_scalars(&mut out, self);
        put_count(&mut out, self.stakes.len());
        for (owner, position) in &self.stakes {
            out.extend_from_slice(owner);
            put_position(&mut out, position);
        }
        put_count(&mut out, self.balances.len());
        for (owner, balance) in &self.balances {
            out.extend_from_slice(owner);
            out.extend_from_slice(&balance.to_le_bytes());
        }
        put_pending(&mut out, &self.guardian_pending);
        out.extend_from_slice(&self.guardian_balance.to_le_bytes());
        out.extend_from_slice(&self.next_entry_id.to_le_bytes());
        put_count(&mut out, self.entries.len());
        for (id, entry) in &self.entries {
            out.extend_from_slice(&id.to_le_bytes());
            put_entry(&mut out, entry);
        }
        out
    }

    /// Decode the canonical bytes, refusing any other.
    pub fn decode(bytes: &[u8]) -> Result<Self, GuardianError> {
        let mut r = Reader(bytes);
        if r.u8()? != GUARDIAN_STATE_VERSION {
            return Err(BAD);
        }
        let mut s = GuardianState::default();
        read_scalars(&mut r, &mut s)?;
        let mut last: Option<[u8; 32]> = None;
        for _ in 0..r.u32()? {
            let owner = r.key()?;
            let position = read_position(&mut r)?;
            if last.is_some_and(|l| l >= owner)
                || (position.active == 0 && position.activating == 0)
            {
                return Err(BAD);
            }
            last = Some(owner);
            s.stakes.insert(owner, position);
        }
        last = None;
        for _ in 0..r.u32()? {
            let owner = r.key()?;
            let balance = r.u64()?;
            if last.is_some_and(|l| l >= owner) || balance == 0 {
                return Err(BAD);
            }
            last = Some(owner);
            s.balances.insert(owner, balance);
        }
        s.guardian_pending = read_pending(&mut r)?;
        s.guardian_balance = r.u64()?;
        s.next_entry_id = r.u64()?;
        let mut last_id: Option<u64> = None;
        for _ in 0..r.u32()? {
            let id = r.u64()?;
            let entry = read_entry(&mut r)?;
            if last_id.is_some_and(|l| l >= id) || entry.amount == 0 || id >= s.next_entry_id {
                return Err(BAD);
            }
            last_id = Some(id);
            s.entries.insert(id, entry);
        }
        r.finish()?;
        Ok(s)
    }
}

/// The undo of one block: the state's scalars and pending shares before
/// it, and the previous value (or absence) of every stake, balance and
/// entry key the block changed. Kept for the last `UNDO_RETENTION_DEPTH`
/// heights; reverted in reverse order down to a fork point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardianUndo {
    scalars: GuardianState,
    stakes: Vec<([u8; 32], Option<StakePosition>)>,
    balances: Vec<([u8; 32], Option<u64>)>,
    entries: Vec<(u64, Option<PayoutEntry>)>,
}

/// The keys whose value differs between `before` and `after`, with their
/// value in `before`.
fn touched<K: Ord + Copy, V: Clone + PartialEq>(
    before: &BTreeMap<K, V>,
    after: &BTreeMap<K, V>,
) -> Vec<(K, Option<V>)> {
    let mut out: Vec<(K, Option<V>)> = before
        .iter()
        .filter(|(k, v)| after.get(k) != Some(v))
        .map(|(k, v)| (*k, Some(v.clone())))
        .collect();
    out.extend(
        after
            .keys()
            .filter(|k| !before.contains_key(k))
            .map(|k| (*k, None)),
    );
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn restore<K: Ord + Copy, V: Clone>(map: &mut BTreeMap<K, V>, keys: &[(K, Option<V>)]) {
    for (key, value) in keys {
        match value {
            Some(value) => {
                map.insert(*key, value.clone());
            }
            None => {
                map.remove(key);
            }
        }
    }
}

impl GuardianUndo {
    /// The undo of the block that took the state from `before` to `after`.
    pub fn between(before: &GuardianState, after: &GuardianState) -> Self {
        let mut scalars = before.clone();
        scalars.stakes.clear();
        scalars.balances.clear();
        scalars.entries.clear();
        Self {
            scalars,
            stakes: touched(&before.stakes, &after.stakes),
            balances: touched(&before.balances, &after.balances),
            entries: touched(&before.entries, &after.entries),
        }
    }

    /// The state before the block, from the state after it.
    pub fn revert(&self, after: &GuardianState) -> GuardianState {
        let mut before = self.scalars.clone();
        before.stakes = after.stakes.clone();
        before.balances = after.balances.clone();
        before.entries = after.entries.clone();
        restore(&mut before.stakes, &self.stakes);
        restore(&mut before.balances, &self.balances);
        restore(&mut before.entries, &self.entries);
        before
    }

    /// How many table keys the block changed.
    pub fn touched_keys(&self) -> usize {
        self.stakes.len() + self.balances.len() + self.entries.len()
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = vec![GUARDIAN_STATE_VERSION];
        put_scalars(&mut out, &self.scalars);
        put_pending(&mut out, &self.scalars.guardian_pending);
        out.extend_from_slice(&self.scalars.guardian_balance.to_le_bytes());
        out.extend_from_slice(&self.scalars.next_entry_id.to_le_bytes());
        put_count(&mut out, self.stakes.len());
        for (owner, position) in &self.stakes {
            out.extend_from_slice(owner);
            match position {
                None => out.push(0),
                Some(p) => {
                    out.push(1);
                    put_position(&mut out, p);
                }
            }
        }
        put_count(&mut out, self.balances.len());
        for (owner, balance) in &self.balances {
            out.extend_from_slice(owner);
            put_option(&mut out, *balance);
        }
        put_count(&mut out, self.entries.len());
        for (id, entry) in &self.entries {
            out.extend_from_slice(&id.to_le_bytes());
            match entry {
                None => out.push(0),
                Some(e) => {
                    out.push(1);
                    put_entry(&mut out, e);
                }
            }
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, GuardianError> {
        let mut r = Reader(bytes);
        if r.u8()? != GUARDIAN_STATE_VERSION {
            return Err(BAD);
        }
        let mut scalars = GuardianState::default();
        read_scalars(&mut r, &mut scalars)?;
        scalars.guardian_pending = read_pending(&mut r)?;
        scalars.guardian_balance = r.u64()?;
        scalars.next_entry_id = r.u64()?;
        let mut stakes = Vec::new();
        for _ in 0..r.u32()? {
            let owner = r.key()?;
            let value = match r.u8()? {
                0 => None,
                1 => Some(read_position(&mut r)?),
                _ => return Err(BAD),
            };
            stakes.push((owner, value));
        }
        let mut balances = Vec::new();
        for _ in 0..r.u32()? {
            let owner = r.key()?;
            balances.push((owner, r.option()?));
        }
        let mut entries = Vec::new();
        for _ in 0..r.u32()? {
            let id = r.u64()?;
            let value = match r.u8()? {
                0 => None,
                1 => Some(read_entry(&mut r)?),
                _ => return Err(BAD),
            };
            entries.push((id, value));
        }
        r.finish()?;
        Ok(Self {
            scalars,
            stakes,
            balances,
            entries,
        })
    }
}

/// One object of the snapshot journal, in chain order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuardianJournalEntry {
    pub height: u64,
    pub kind: GuardianObjectKind,
    pub body: Vec<u8>,
}

/// `magic ‖ version ‖ count:u64 ‖ count × { height:u64 ‖ kind:u8 ‖ len:u32 ‖ body }`.
pub fn encode_guardian_journal(entries: &[GuardianJournalEntry]) -> Vec<u8> {
    let mut out = GUARDIAN_JOURNAL_MAGIC.to_vec();
    out.push(GUARDIAN_JOURNAL_VERSION);
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        out.extend_from_slice(&entry.height.to_le_bytes());
        out.push(entry.kind.section_code());
        out.extend_from_slice(
            &u32::try_from(entry.body.len())
                .expect("body fits u32")
                .to_le_bytes(),
        );
        out.extend_from_slice(&entry.body);
    }
    out
}

/// Decode a journal file: known kinds at their exact sizes, heights never
/// decreasing, within [`MAX_GUARDIAN_JOURNAL_FILE_BYTES`], nothing after.
pub fn decode_guardian_journal(bytes: &[u8]) -> Result<Vec<GuardianJournalEntry>, GuardianError> {
    if bytes.len() > MAX_GUARDIAN_JOURNAL_FILE_BYTES {
        return Err(BAD);
    }
    let mut r = Reader(bytes);
    if r.take(8)? != GUARDIAN_JOURNAL_MAGIC || r.u8()? != GUARDIAN_JOURNAL_VERSION {
        return Err(BAD);
    }
    let count = r.u64()?;
    let mut entries: Vec<GuardianJournalEntry> = Vec::new();
    for _ in 0..count {
        let height = r.u64()?;
        let kind = GuardianObjectKind::from_section_code(r.u8()?).ok_or(BAD)?;
        let len = r.u32()? as usize;
        if !super::guardian_body_len_allowed(kind, len)
            || entries.last().is_some_and(|last| last.height > height)
        {
            return Err(BAD);
        }
        entries.push(GuardianJournalEntry {
            height,
            kind,
            body: r.take(len)?.to_vec(),
        });
    }
    r.finish()?;
    Ok(entries)
}
