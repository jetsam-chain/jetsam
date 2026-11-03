// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-observe`: what the guardian's OWN node shows, and the policy rules
//! that need it (design §3: SG-1 to SG-9 and SG-17). The chain work, the
//! block ids, the digest and the anchor tree are recomputed from the raw
//! headers (`chain`); the node's counters are never an input of a decision.
//! Every instant is the guardian's own clock, never a header timestamp.

use std::collections::BTreeMap;

use std::cmp::Ordering;

use jetsam_chain::consensus::difficulty::block_work;

use crate::chain::{empty_root, Chain, HeaderInfo};
use crate::signer::{kind_of, StatementKind, SIGN_DEPTH_FLOOR};
use crate::{u256, Domain, GuardianStatement};

/// Heights below the tip whose ids the observer re-reads each poll (reorgs).
const REORG_SCAN: u64 = 64;

/// SG-4 on the public networks, mainnet and public test network alike (A4-3,
/// D66): `stability_secs` below this refuses to start. The default is this
/// same value; the private rehearsal is not bound.
pub const STABILITY_SECS_FLOOR: u64 = 1_200;

/// `stability_secs` respects the floor of the profile of `domain`.
pub fn stability_floor_ok(domain: &Domain, stability_secs: u64) -> bool {
    !crate::Profile::of(domain).strict() || stability_secs >= STABILITY_SECS_FLOOR
}

/// The policy thresholds (§3.1), all configurable, the design's defaults.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Thresholds {
    /// SG-1: a heavier branch seen in this window refuses (s).
    pub branch_window: u64,
    /// SG-2: the rate window (s) and its longest span.
    pub rate_window: u64,
    pub rate_max_span: u64,
    /// SG-2: `β = rate_num / rate_den` and the target block time (s).
    pub rate_num: u64,
    pub rate_den: u64,
    pub block_time: u64,
    /// SG-3.
    pub sign_depth: u64,
    /// SG-4: continuous observation of L−7..L (s) and the longest gap (s).
    pub stability_secs: u64,
    pub observation_gap: u64,
    /// SG-5.
    pub min_dispatchable_peers: usize,
    pub tip_max_age: u64,
    pub tip_max_drift: u64,
    /// SG-6.
    pub reorg_window: u64,
    pub reorg_max_depth: u64,
    /// SG-7: peers announcing `v1.7=J` with a verified profile, once J is reached.
    pub min_v17_peers: usize,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            branch_window: 7_200,
            rate_window: 7_200,
            rate_max_span: 14_400,
            rate_num: 4,
            rate_den: 10,
            block_time: 180,
            sign_depth: 8,
            stability_secs: 1_200,
            observation_gap: 120,
            min_dispatchable_peers: 8,
            tip_max_age: 900,
            tip_max_drift: 120,
            reorg_window: 3_600,
            reorg_max_depth: 2,
            min_v17_peers: 4,
        }
    }
}

/// The observer's configuration.
#[derive(Clone, Debug)]
pub struct ObserverConfig {
    pub domain: Domain,
    pub thresholds: Thresholds,
    /// The exact version string the node must report (SG-7).
    pub version: String,
    pub v1_7_height: u64,
    pub deposits_from_height: u64,
    pub genesis_locked_height: Option<u64>,
    pub genesis_go: Option<(u64, [u8; 32])>,
    /// P: the highest pin of the guardian's node (rule (o) of SG-1).
    pub last_pin_height: u64,
    /// Attestation-only statements at the multiples of this height (§2.5).
    pub attest_every: u64,
    pub attest_from: u64,
}

/// SG-5 facts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Health {
    pub synced: bool,
    pub p2p_healthy: bool,
    pub dispatchable_peers: usize,
}

/// One peer's announced version (SG-7).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PeerVersion {
    pub agent: String,
    pub profile_verified: bool,
    pub connected: bool,
}

/// One branch of `getObservedBranches`, its headers decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Branch {
    pub fork_height: u64,
    pub fork_base_id: [u8; 32],
    pub tip_height: u64,
    pub headers_valid: bool,
    /// `accepted`, `pin_mismatch`, `header_rule`, `deeper_than_finality`, `pending`.
    pub verdict: String,
    pub below_last_pin: bool,
    pub reorg_depth: Option<u64>,
    /// The node's own figure (LE): used only when the headers are truncated.
    pub node_work: [u8; 32],
    pub headers: Vec<HeaderInfo>,
    pub headers_truncated: bool,
    pub first_seen: u64,
    pub last_seen: u64,
    pub canonical_height_at_last_seen: u64,
}

/// `getObservedBranches`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Branches {
    pub recording_since: u64,
    pub last_pin_height: Option<u64>,
    pub branches: Vec<Branch>,
}

/// Facts read from the node at one poll, besides the chain.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Facts {
    /// `getBlockHash(0)`.
    pub genesis_id: [u8; 32],
    pub health: Health,
    pub local_version: String,
    pub peers: Vec<PeerVersion>,
    pub branches: Option<Branches>,
    /// `getGuardianState`: key set, revoked at.
    pub group_key_set: bool,
    pub revoked_at: Option<u64>,
    /// `getVaultState.deposits_by_height`.
    pub deposits_by_height: Vec<(u64, u32)>,
}

/// What the observer reads from its node.
pub trait NodeSource {
    fn tip_height(&mut self) -> anyhow::Result<u64>;
    fn block_hash(&mut self, height: u64) -> anyhow::Result<Option<[u8; 32]>>;
    fn header(&mut self, height: u64) -> anyhow::Result<Option<HeaderInfo>>;
    fn facts(&mut self) -> anyhow::Result<Facts>;
    /// The node's own chain work at a height, if it says (diagnostic only:
    /// an alarm when it differs from the recomputed one, never a decision).
    fn chain_work(&mut self, height: u64) -> anyhow::Result<Option<[u8; 32]>> {
        let _ = height;
        Ok(None)
    }
}

/// SG-2: `(W(t) − W(t_s)) × block_time × den ≥ num × block_work(target_s) × (t − t_s)`.
pub fn rate_ok(w_now: &[u8; 32], w_s: &[u8; 32], target_s: &[u8; 32], elapsed: u64, th: &Thresholds) -> bool {
    let gained = u256::sub(w_now, w_s);
    let expected_unit = block_work(target_s);
    u256::mul_ge(&gained, th.block_time * th.rate_den, &expected_unit, th.rate_num * elapsed)
}

/// The observer's verdict on one statement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub ok: bool,
    pub code: String,
}

impl Outcome {
    fn ok() -> Self {
        Self { ok: true, code: "OK".into() }
    }
    fn no(code: impl Into<String>) -> Self {
        Self { ok: false, code: code.into() }
    }
}

/// The observer's state.
pub struct Observer {
    pub cfg: ObserverConfig,
    chain: Chain,
    /// `height → (id, first seen)` under continuous observation.
    first_seen: BTreeMap<u64, ([u8; 32], u64)>,
    last_poll: Option<u64>,
    /// `(t, W(tip), target(tip))`, at least one per minute.
    samples: Vec<(u64, [u8; 32], [u8; 32])>,
    /// `(t, depth)` of the reorganizations seen.
    reorgs: Vec<(u64, u64)>,
    tip_changed_at: u64,
    facts: Facts,
    deposits: BTreeMap<u64, u32>,
    /// A node anomaly (a header that does not link, an id that differs).
    pub alarm: Option<String>,
}

impl Observer {
    /// A new observer over `chain` (its trust anchor); refuses
    /// `sign_depth < 8`.
    pub fn new(cfg: ObserverConfig, chain: Chain) -> anyhow::Result<Self> {
        anyhow::ensure!(
            cfg.thresholds.sign_depth >= SIGN_DEPTH_FLOOR,
            "sign_depth {} is below the floor {SIGN_DEPTH_FLOOR}: the observer does not start",
            cfg.thresholds.sign_depth
        );
        anyhow::ensure!(
            stability_floor_ok(&cfg.domain, cfg.thresholds.stability_secs),
            "stability_secs {} is below the floor {STABILITY_SECS_FLOOR} of {} (SG-4, D66): the observer does not start",
            cfg.thresholds.stability_secs,
            crate::Profile::of(&cfg.domain)
        );
        anyhow::ensure!(cfg.attest_every > 0, "attest_every must be positive");
        Ok(Self {
            cfg,
            chain,
            first_seen: BTreeMap::new(),
            last_poll: None,
            samples: Vec::new(),
            reorgs: Vec::new(),
            tip_changed_at: 0,
            facts: Facts::default(),
            deposits: BTreeMap::new(),
            alarm: None,
        })
    }

    /// The observer's chain.
    pub fn chain(&self) -> &Chain {
        &self.chain
    }

    /// `(time, depth)` of the reorganizations seen (kept twice `reorg_window`; `jg-fill`'s guard).
    pub fn reorgs(&self) -> &[(u64, u64)] {
        &self.reorgs
    }

    /// When the tip last changed, on this program's clock.
    pub fn tip_changed_at(&self) -> u64 {
        self.tip_changed_at
    }

    /// The facts of the last poll (health, branches, …).
    pub fn facts(&self) -> &Facts {
        &self.facts
    }

    /// One poll of the node at `now`.
    pub fn poll(&mut self, now: u64, node: &mut dyn NodeSource) -> anyhow::Result<()> {
        let old_tip = self.chain.tip_height();
        let old_tip_id = self.chain.tip_id();
        let node_tip = node.tip_height()?;
        // the highest height both chains share, scanning down from the lower tip
        let mut common = old_tip.min(node_tip);
        let floor = self.chain.base_height();
        let mut scanned = 0u64;
        loop {
            match node.block_hash(common)? {
                Some(id) if Some(id) == self.chain.id_at(common) => break,
                _ => {
                    if common == floor {
                        self.alarm = Some(format!("NODE_ANOMALY: the node's block at the trust anchor {floor} differs"));
                        return Ok(());
                    }
                    common -= 1;
                    scanned += 1;
                    if scanned > 10 * REORG_SCAN {
                        self.alarm = Some("NODE_ANOMALY: a reorganization deeper than the scan".into());
                        return Ok(());
                    }
                }
            }
        }
        if common < old_tip {
            self.reorgs.push((now, old_tip - common));
            self.chain.truncate_above(common);
            self.first_seen.retain(|h, _| *h <= common);
        }
        let mut clean = true;
        for h in common + 1..=node_tip {
            let Some(header) = node.header(h)? else {
                break;
            };
            if node.block_hash(h)? != Some(header.id) {
                self.alarm = Some(format!("NODE_ANOMALY: getBlockHash({h}) differs from the id recomputed from its header"));
                clean = false;
                break;
            }
            if let Err(e) = self.chain.extend(header) {
                self.alarm = Some(format!("NODE_ANOMALY: {e}"));
                clean = false;
                break;
            }
        }
        let tip = self.chain.tip_height();
        if let Some(w) = node.chain_work(tip)? {
            if Some(w) != self.chain.work_at(tip) {
                // diagnostic only: the recomputed work decides
                tracing_like(&format!("node chain work differs at {tip} (diagnostic)"));
            }
        }
        if clean && self.alarm.as_deref().is_some_and(|a| a.starts_with("NODE_ANOMALY")) {
            self.alarm = None;
        }
        // tip age, on the guardian's clock
        if self.chain.tip_id() != old_tip_id || self.last_poll.is_none() {
            self.tip_changed_at = now;
        }
        // continuous observation: a gap restarts every first-seen time
        if self.last_poll.is_some_and(|t| now.saturating_sub(t) > self.cfg.thresholds.observation_gap) {
            for v in self.first_seen.values_mut() {
                v.1 = now;
            }
        }
        let low = tip.saturating_sub(REORG_SCAN * 16).max(self.chain.base_height());
        for h in low..=tip {
            let id = self.chain.id_at(h).expect("within the chain");
            match self.first_seen.get(&h) {
                Some((seen, _)) if *seen == id => {}
                _ => {
                    self.first_seen.insert(h, (id, now));
                }
            }
        }
        self.first_seen.retain(|h, _| *h >= low);
        // rate samples, at most one per minute, on the guardian's clock
        if self.samples.last().is_none_or(|(t, _, _)| now >= t + 60) {
            let target = self.chain.header_at(tip).map(|h| h.target).unwrap_or([0xffu8; 32]);
            self.samples.push((now, self.chain.work_at(tip).expect("tip"), target));
            let keep_from = now.saturating_sub(self.cfg.thresholds.rate_max_span + 3_600);
            self.samples.retain(|(t, _, _)| *t >= keep_from);
        }
        self.reorgs.retain(|(t, _)| *t + self.cfg.thresholds.reorg_window * 2 >= now);
        self.last_poll = Some(now);
        let facts = node.facts()?;
        self.set_facts(facts);
        Ok(())
    }

    /// Set the facts directly (tests, and the poll).
    pub fn set_facts(&mut self, facts: Facts) {
        for (h, n) in &facts.deposits_by_height {
            self.deposits.insert(*h, *n);
        }
        self.facts = facts;
    }

    /// SG-1 to SG-9 and SG-17 for `s` at `now`.
    pub fn evaluate(&self, s: &GuardianStatement, now: u64) -> Outcome {
        let th = &self.cfg.thresholds;
        let kind = kind_of(s);
        if let Some(a) = &self.alarm {
            return Outcome::no(a.clone());
        }
        // SG-7: network and version
        let tip = self.chain.tip_height();
        let v17_peers = self
            .facts
            .peers
            .iter()
            .filter(|p| p.connected && p.profile_verified && p.agent.contains(&format!("v1.7={}", self.cfg.v1_7_height)))
            .count();
        if self.facts.genesis_id != self.cfg.domain.network
            || self.facts.local_version != self.cfg.version
            || (tip + 1 >= self.cfg.v1_7_height && v17_peers < th.min_v17_peers)
        {
            return Outcome::no("WRONG_NETWORK_OR_VERSION");
        }
        // SG-5: health
        let h = &self.facts.health;
        if !h.synced {
            return Outcome::no("NODE_UNHEALTHY:not_synced");
        }
        if !h.p2p_healthy {
            return Outcome::no("NODE_UNHEALTHY:p2p");
        }
        if h.dispatchable_peers < th.min_dispatchable_peers {
            return Outcome::no("NODE_UNHEALTHY:peers");
        }
        if now.saturating_sub(self.tip_changed_at) > th.tip_max_age {
            return Outcome::no("NODE_UNHEALTHY:tip_age");
        }
        let tip_ts = self.chain.header_at(tip).map(|x| x.timestamp).unwrap_or(0);
        if tip_ts > now + th.tip_max_drift {
            return Outcome::no("NODE_UNHEALTHY:tip_in_the_future");
        }
        // SG-9: the key (the authorized genesis is the single exception)
        let authorized_genesis = kind == StatementKind::Genesis && self.cfg.genesis_go == Some((s.height, s.block_id));
        if (!self.facts.group_key_set || self.facts.revoked_at.is_some()) && !authorized_genesis {
            return Outcome::no("KEY_REVOKED");
        }
        // SG-3: final on the node, measured above the block described
        if s.height.saturating_add(th.sign_depth) > tip {
            return Outcome::no("NOT_FINAL");
        }
        // SG-8: exactness, recomputed from the headers
        match self.check_fields(s, kind) {
            Some(field) => return Outcome::no(format!("FIELD_MISMATCH:{field}")),
            None => {}
        }
        // SG-4: L−7..L unchanged under continuous observation
        if !self.point_stable(s.height, now) {
            return Outcome::no("POINT_NOT_STABLE");
        }
        // SG-6: no reorganization deeper than 2 in the last hour
        let recent_deep = self.reorgs.iter().any(|(t, d)| *t + th.reorg_window >= now && *d > th.reorg_max_depth)
            || self.facts.branches.as_ref().is_some_and(|b| {
                b.branches.iter().any(|x| {
                    x.verdict == "accepted" && x.reorg_depth.is_some_and(|d| d > th.reorg_max_depth) && x.last_seen + th.reorg_window >= now
                })
            });
        if recent_deep {
            return Outcome::no("RECENT_DEEP_REORG");
        }
        // SG-1: no heavier branch seen in the window
        if let Some(code) = self.heavier_branch(now) {
            return Outcome::no(code);
        }
        // SG-2: work rate over the window, on the guardian's clock
        let Some((t_s, w_s, target_s)) = self
            .samples
            .iter()
            .rev()
            .find(|(t, _, _)| *t + th.rate_window <= now)
            .copied()
            .filter(|(t, _, _)| now - t <= th.rate_max_span)
        else {
            return Outcome::no("RATE_WINDOW_NOT_COVERED");
        };
        let w_now = self.chain.work_at(tip).expect("tip");
        if !rate_ok(&w_now, &w_s, &target_s, now - t_s, th) {
            return Outcome::no("WORK_RATE_LOW");
        }
        // SG-17: deposits no more than this node counted
        if kind == StatementKind::Cosign && s.deposit_count > self.deposits_up_to(s.height) {
            return Outcome::no("DEPOSIT_COUNT_IMPLAUSIBLE");
        }
        Outcome::ok()
    }

    /// SG-4: heights `height − 7 ..= height` seen with their current ids, each
    /// for at least `stability_secs` of continuous observation.
    /// Below 7 there is no such window: never stable (A4-7).
    fn point_stable(&self, height: u64, now: u64) -> bool {
        let th = &self.cfg.thresholds;
        let Some(low) = height.checked_sub(7) else {
            return false;
        };
        for x in low..=height {
            match self.first_seen.get(&x) {
                Some((id, t)) if Some(*id) == self.chain.id_at(x) && now >= t + th.stability_secs => {}
                _ => return false,
            }
        }
        true
    }

    /// SG-8: the first field that differs from the chain, if any.
    fn check_fields(&self, s: &GuardianStatement, kind: StatementKind) -> Option<&'static str> {
        if !self.cfg.domain.matches(s) {
            return Some("domain");
        }
        if self.chain.id_at(s.height) != Some(s.block_id) {
            return Some("blockId");
        }
        if self.chain.work_at(s.height).map(|w| u256::to_be(&w)) != Some(s.cum_work) {
            return Some("cumWork");
        }
        if self.chain.headers_digest(s.height) != Some(s.headers_digest) {
            return Some("headersDigest");
        }
        match kind {
            StatementKind::AttestationOnly => None,
            StatementKind::Genesis => (s.anchor_root != empty_root()).then_some("anchorRoot"),
            StatementKind::Cosign => {
                let gl = self.cfg.genesis_locked_height?;
                (self.chain.anchor_root(gl, s.height) != Some(s.anchor_root)).then_some("anchorRoot")
            }
        }
    }

    /// SG-1 with rule (o): a code if a counted branch is heavier.
    fn heavier_branch(&self, now: u64) -> Option<String> {
        let th = &self.cfg.thresholds;
        let Some(b) = &self.facts.branches else {
            return Some("BRANCH_WINDOW_NOT_COVERED".into());
        };
        if now < b.recording_since + th.branch_window {
            return Some("BRANCH_WINDOW_NOT_COVERED".into());
        }
        let pin = self.cfg.last_pin_height.max(b.last_pin_height.unwrap_or(0));
        for br in &b.branches {
            if br.last_seen + th.branch_window < now || !br.headers_valid || br.verdict == "accepted" {
                continue;
            }
            // rule (o): a branch forking below the last pin is never counted
            if br.fork_height < pin {
                continue;
            }
            let work = match (self.chain.work_at(br.fork_height), self.branch_links(br)) {
                (Some(base), true) if !br.headers_truncated => {
                    br.headers.iter().fold(base, |w, h| u256::add(&w, &h.work))
                }
                // the node's figure: only in the safe direction (it can refuse)
                _ => br.node_work,
            };
            let canonical = self
                .chain
                .work_at(br.canonical_height_at_last_seen)
                .unwrap_or_else(|| self.chain.work_at(self.chain.tip_height()).expect("tip"));
            if u256::cmp(&work, &canonical) == Ordering::Greater {
                return Some("HEAVIER_BRANCH_SEEN".into());
            }
        }
        None
    }

    fn branch_links(&self, br: &Branch) -> bool {
        let mut prev = br.fork_base_id;
        if self.chain.id_at(br.fork_height) != Some(prev) {
            return false;
        }
        for (i, h) in br.headers.iter().enumerate() {
            if h.prev != prev || h.height != br.fork_height + 1 + i as u64 {
                return false;
            }
            prev = h.id;
        }
        true
    }

    /// The attestation-only statements admissible now (multiples of
    /// `attest_every` at or above `attest_from`, the latest first, at most 2).
    pub fn attestations_due(&self, now: u64) -> Vec<GuardianStatement> {
        let tip = self.chain.tip_height();
        let Some(top) = tip.checked_sub(self.cfg.thresholds.sign_depth) else {
            return Vec::new();
        };
        let every = self.cfg.attest_every;
        let mut out = Vec::new();
        let mut h = top - top % every;
        while out.len() < 2 && h >= self.cfg.attest_from.max(self.chain.base_height() + 7) && h > 0 {
            if let Some(s) = crate::chain::statement_at(&self.cfg.domain, &self.chain, h, crate::chain::Form::AttestationOnly) {
                if self.evaluate(&s, now).ok {
                    out.push(s);
                }
            }
            if h < every {
                break;
            }
            h -= every;
        }
        out
    }

    /// The observation state worth keeping across a restart (rate samples,
    /// first-seen times, reorganizations, deposit counts): without it, a
    /// restarted `jg-observe` refuses for 2 h (`RATE_WINDOW_NOT_COVERED`).
    pub fn export_state(&self) -> serde_json::Value {
        serde_json::json!({
            "last_poll": self.last_poll,
            "tip_changed_at": self.tip_changed_at,
            "samples": self.samples.iter().map(|(t, w, x)| (t, hex::encode(w), hex::encode(x))).collect::<Vec<_>>(),
            "reorgs": self.reorgs,
            "first_seen": self.first_seen.iter().map(|(h, (id, t))| (h, hex::encode(id), t)).collect::<Vec<_>>(),
            "deposits": self.deposits.iter().collect::<Vec<_>>(),
        })
    }

    /// Restore [`Self::export_state`]; the next poll applies the gap rule.
    pub fn import_state(&mut self, v: &serde_json::Value) -> anyhow::Result<()> {
        self.last_poll = v["last_poll"].as_u64();
        self.tip_changed_at = v["tip_changed_at"].as_u64().unwrap_or(0);
        self.samples.clear();
        for e in v["samples"].as_array().cloned().unwrap_or_default() {
            self.samples.push((e[0].as_u64().unwrap_or(0), crate::parse32(e[1].as_str().unwrap_or_default())?, crate::parse32(e[2].as_str().unwrap_or_default())?));
        }
        self.reorgs = serde_json::from_value(v["reorgs"].clone()).unwrap_or_default();
        self.first_seen.clear();
        for e in v["first_seen"].as_array().cloned().unwrap_or_default() {
            self.first_seen.insert(e[0].as_u64().unwrap_or(0), (crate::parse32(e[1].as_str().unwrap_or_default())?, e[2].as_u64().unwrap_or(0)));
        }
        for e in v["deposits"].as_array().cloned().unwrap_or_default() {
            self.deposits.insert(e[0].as_u64().unwrap_or(0), e[1].as_u64().unwrap_or(0) as u32);
        }
        Ok(())
    }

    /// The `(x, id)` of the last `n` final heights and the tip (feed `obs`).
    pub fn final_ids(&self, n: u64) -> (u64, Vec<(u64, [u8; 32])>) {
        let tip = self.chain.tip_height();
        let top = tip.saturating_sub(self.cfg.thresholds.sign_depth);
        let low = top.saturating_sub(n.saturating_sub(1)).max(self.chain.base_height());
        (tip, (low..=top).filter_map(|x| self.chain.id_at(x).map(|id| (x, id))).collect())
    }

    /// The cumulative V4 deposits of `]F ; height]` this node counted.
    pub fn deposits_up_to(&self, height: u64) -> u64 {
        if height <= self.cfg.deposits_from_height {
            return 0;
        }
        self.deposits
            .range(self.cfg.deposits_from_height + 1..=height)
            .map(|(_, n)| u64::from(*n))
            .sum()
    }
}

fn tracing_like(_message: &str) {}

#[cfg(test)]
#[path = "observer_tests.rs"]
mod tests;
