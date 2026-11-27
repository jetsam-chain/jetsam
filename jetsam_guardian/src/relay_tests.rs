use super::*;
use crate::observer::Thresholds;
use crate::signer::SignerConfig;
use crate::Domain;

const T0: u64 = 1_800_000_000;

fn id(l: u64) -> [u8; 32] {
    let mut x = [0u8; 32];
    x[..8].copy_from_slice(&l.to_be_bytes());
    x
}

fn name(l: u64, not_after: u64) -> String {
    format!("cp-{l:08}-{not_after}")
}

fn live(l: u64, not_after: u64) -> Live {
    Live { name: name(l, not_after), id: id(l), not_after }
}

/// Points L = 100..=110, one block every 90 s: L = 100 first seen at T0.
fn fast_points() -> Vec<Point> {
    (100..=110).map(|l| Point { id: id(l), locked_height: l, locked_seen_at: T0 + 90 * (l - 100), on_chain: true }).collect()
}

fn dropped(plan: &Plan) -> Vec<String> {
    let mut d = plan.drop.clone();
    d.sort();
    d
}

#[test]
fn relay_offers_again_the_highest_already_stable_point_when_no_proposal_of_it_is_left() {
    // stable for the relayer: seen + 1 200 + 15 <= now, so L <= 103 at T0 + 1 500
    let now = T0 + 1_500;
    let l1 = now + 40;
    let lv: Vec<Live> = (108..=110).map(|l| live(l, l1 + 2_900)).collect();
    let plan = plan(&fast_points(), &lv, 99, now, l1, &Policy::default());
    assert_eq!(plan.propose, Some(id(103)));
    assert!(plan.drop.is_empty(), "the newest points stay: they become stable later");
}

#[test]
fn relay_does_not_offer_a_point_younger_than_sg4_and_its_margin() {
    let p = Policy::default();
    let at = T0 + p.stability_secs + p.margin_secs;
    let lv = vec![live(110, at + 2_900)];
    assert_eq!(plan(&fast_points(), &lv, 99, at - 1, at, &p).propose, None);
    assert_eq!(plan(&fast_points(), &lv, 99, at, at, &p).propose, Some(id(100)));
}

#[test]
fn relay_keeps_every_point_that_can_still_become_final() {
    let now = T0 + 1_000;
    let lv: Vec<Live> = (100..=110).map(|l| live(l, now + 2_000)).collect();
    let plan = plan(&fast_points(), &lv, 99, now, now, &Policy::default());
    assert!(plan.drop.is_empty(), "a pending point is never dropped while it can become final: {:?}", plan.drop);
}

#[test]
fn relay_drops_proposals_at_or_below_the_final_point() {
    let now = T0 + 600;
    let lv: Vec<Live> = (100..=105).map(|l| live(l, now + 2_000)).collect();
    let plan = plan(&fast_points(), &lv, 103, now, now, &Policy::default());
    assert_eq!(dropped(&plan), (100..=103).map(|l| name(l, now + 2_000)).collect::<Vec<_>>());
    assert_eq!(plan.propose, None);
}

#[test]
fn relay_drops_proposals_whose_not_after_is_no_longer_signable() {
    let p = Policy::default();
    let now = T0 + 600;
    let l1 = now + 50;
    let lv = vec![live(105, l1 + p.not_after_min - 1), live(106, l1 + p.not_after_min)];
    let plan = plan(&fast_points(), &lv, 99, now, l1, &p);
    assert_eq!(plan.drop, vec![name(105, l1 + p.not_after_min - 1)]);
}

#[test]
fn relay_offers_again_a_stable_point_whose_proposals_leave_too_little_time() {
    let p = Policy::default();
    let now = T0 + 1_500;
    let l1 = now + 40;
    let short = vec![live(103, l1 + p.renew_before_secs - 1)];
    let renewed = plan(&fast_points(), &short, 99, now, l1, &p);
    assert_eq!(renewed.propose, Some(id(103)));
    assert!(renewed.drop.is_empty(), "the old proposal stays while it is signable: partials may be on it");
    let enough = vec![live(103, l1 + p.renew_before_secs)];
    assert_eq!(plan(&fast_points(), &enough, 99, now, l1, &p).propose, None);
}

#[test]
fn relay_never_offers_or_keeps_a_point_off_its_chain() {
    let now = T0 + 1_500;
    let mut points = fast_points();
    points[3].on_chain = false; // L = 103
    let lv = vec![live(103, now + 2_900), live(110, now + 2_900), Live { name: "stray".into(), id: [9u8; 32], not_after: now + 2_900 }];
    let plan = plan(&points, &lv, 99, now, now, &Policy::default());
    assert_eq!(dropped(&plan), vec![name(103, now + 2_900), "stray".to_string()]);
    assert_eq!(plan.propose, Some(id(102)));
}

#[test]
fn relay_bounds_its_proposals_dropping_the_lowest_never_the_best() {
    let p = Policy { max_live: 4, ..Policy::default() };
    let now = T0 + 1_500; // the best stable point is 103
    let na = now + 2_900;
    let lv: Vec<Live> = (101..=107).map(|l| live(l, na)).collect();
    let kept = plan(&fast_points(), &lv, 99, now, now, &p);
    assert_eq!(kept.propose, None);
    assert_eq!(dropped(&kept), vec![name(101, na), name(102, na), name(104, na)]);
    // when the best one is offered again it takes a place too
    let lv: Vec<Live> = [101, 102, 104, 105, 106, 107].iter().map(|l| live(*l, na)).collect();
    let again = plan(&fast_points(), &lv, 99, now, now, &p);
    assert_eq!(again.propose, Some(id(103)));
    assert_eq!(dropped(&again), vec![name(101, na), name(102, na), name(104, na)]);
}

#[test]
fn relay_policy_follows_the_guardians_rules() {
    let p = Policy::default();
    assert_eq!(p.stability_secs, Thresholds::default().stability_secs, "SG-4 is the guardians' default, unchanged");
    let s = SignerConfig::defaults(Domain { network: [1u8; 32], l1_chain_id: 31_337, l1_bridge: [0xeeu8; 20] }, 96, 128);
    assert_eq!(p.not_after_min, s.not_after_min, "SG-11 lower bound");
    assert!(p.renew_before_secs > p.not_after_min);
    // the relayer's proposals, its genesis proposal and at most 2 attestations due fit one jg-net round
    assert!(p.max_live + 1 + 2 <= crate::NET_ROUND_CANDIDATES);
}

/// A relayer that submits a point when T = L + 8 appears and applies `plan`
/// every 10 s, and guardians that sign every 15 s the highest proposal they
/// accept: SG-4 on their own first sight (`LAG` s after the relayer's), their
/// node at L + 8, `notAfter ≥ now + 300`. Returns the first sights and the
/// finalizations (L, time).
const LAG: u64 = 5;

fn simulate(intervals: &[u64], down: (u64, u64), end: u64) -> (Vec<u64>, Vec<(u64, u64)>) {
    let p = Policy::default();
    let mut seen = vec![0u64];
    let mut next = intervals[0];
    let mut k = 0;
    let (mut points, mut lives): (Vec<Point>, Vec<Live>) = (Vec::new(), Vec::new());
    let (mut last_l, mut final_locked) = (0u64, 0u64);
    let mut finals = Vec::new();
    for now in 0..end {
        while k < intervals.len() && now >= next {
            seen.push(next);
            k += 1;
            next += intervals.get(k).copied().unwrap_or(u64::MAX / 4);
        }
        let tip = (seen.len() - 1) as u64;
        if now % 10 == 0 {
            let l = tip.saturating_sub(8);
            if l > last_l {
                last_l = l;
                points.push(Point { id: id(l), locked_height: l, locked_seen_at: seen[l as usize], on_chain: true });
                lives.push(live(l, now + 3_000));
            }
            let pl = plan(&points, &lives, final_locked, now, now, &p);
            lives.retain(|x| !pl.drop.contains(&x.name));
            if let Some(b) = pl.propose {
                let l = points.iter().find(|x| x.id == b).expect("a submitted point").locked_height;
                lives.push(Live { name: format!("{}-again", name(l, now + 3_000)), id: b, not_after: now + 3_000 });
            }
        }
        if now % 15 == 7 && !(down.0..down.1).contains(&now) {
            let best = lives
                .iter()
                .filter(|x| x.not_after >= now + p.not_after_min)
                .filter_map(|x| points.iter().find(|q| q.id == x.id))
                .map(|q| q.locked_height)
                .filter(|l| *l > final_locked && seen[*l as usize] + LAG + 1_200 <= now && seen[*l as usize + 8] + LAG <= now)
                .max();
            if let Some(l) = best {
                final_locked = l;
                finals.push((l, now));
            }
        }
    }
    (seen, finals)
}

#[test]
fn relay_plan_finalizes_at_the_sg4_floor_with_fast_blocks_and_after_an_outage() {
    // 60 blocks 30 s apart, 20 at 180 s, a stall of 4 500 s, 20 at 180 s; the
    // guardians do not sign from 4 000 to 9 500 (the stall's proposals expire)
    let mut iv = vec![30u64; 60];
    iv.extend([180u64; 20]);
    iv.push(4_500);
    iv.extend([180u64; 20]);
    let down = (4_000, 9_500);
    let (seen, finals) = simulate(&iv, down, 14_500);
    let tip = (seen.len() - 1) as u64;
    assert_eq!(finals.last().map(|f| f.0), Some(tip - 8), "every point ends final: {finals:?}");
    // a deposit at h is final once SOME point L >= h is signed: at the earliest when point h
    // itself is (its L + 8 there, SG-4 on h), plus a relayer step (10 s), a guardian round
    // (15 s) and the relayer's margin (15 s); during the outage, at its end plus a round
    for h in 1..=tip - 8 {
        let hh = h as usize;
        let earliest = (seen[hh + 8] + LAG).max(seen[hh] + LAG + 1_200);
        let allowed = if earliest + 40 >= down.0 && earliest <= down.1 { down.1 + 25 } else { earliest + 40 };
        let covered = finals.iter().find(|f| f.0 >= h).map(|f| f.1).expect("covered");
        assert!(covered <= allowed, "deposit at {h}: final at {covered}, signable from {earliest}, allowed {allowed}: {finals:?}");
    }
    // SG-4 still binds with fast blocks (8 blocks of 30 s): nothing is final before it
    let (l, t) = finals[0];
    assert!(t >= seen[l as usize] + LAG + 1_200 && t - seen[l as usize + 8] >= 1_200 - 8 * 30);
}

// ---------------------------------------------------------------- D-AV3: the checkpoint cadence and MN-3

#[test]
fn the_cadence_is_one_point_every_6600_s_by_default_never_above_w_stall_minus_the_margin() {
    assert_eq!(DEFAULT_CHECKPOINT_CADENCE, 6_600);
    assert_eq!(W_STALL, 7_200, "the contract's W_STALL (JetsamBridge.sol)");
    assert_eq!(DEFAULT_CHECKPOINT_CADENCE, W_STALL - CADENCE_MARGIN_SECS);
    assert!(check_cadence(DEFAULT_CHECKPOINT_CADENCE).is_ok());
    assert!(check_cadence(600).is_ok(), "more often is allowed");
    assert!(check_cadence(W_STALL - CADENCE_MARGIN_SECS + 1).is_err());
    assert!(check_cadence(W_STALL).is_err());
    assert!(check_cadence(0).is_err());
}

#[test]
fn a_checkpoint_is_due_once_the_cadence_has_elapsed_since_the_last_accepted_one() {
    assert!(checkpoint_due(None, 1_000, 6_600), "the first point after the genesis");
    assert!(!checkpoint_due(Some(1_000), 7_599, 6_600));
    assert!(checkpoint_due(Some(1_000), 7_600, 6_600));
}

const GEN: u64 = 1_800_000_000;

fn w(n: u64) -> [u8; 32] {
    crate::u256::from_u64(n)
}

#[test]
fn the_stall_model_is_the_contracts_rule() {
    // unit block work 1000; 7 200 s = 40 blocks at 180 s; stalled below 35 % of them = 14 blocks
    let mut m = StallModel::genesis(GEN, w(1_000_000));
    assert!(!m.stalled(GEN + 100_000), "no submission yet: tipBlockWork is 0, never stalled");
    m.submit(GEN + 100, w(1_000_000 + 14_000), w(1_000));
    assert!(!m.stalled(GEN + 7_199), "never within W_STALL of the genesis");
    // at GEN + 7 200: W(GEN) = 1 000 000 (genesis sample), produced 14 000 = 14 blocks: not < 14
    assert!(!m.stalled(GEN + 7_200));
    // at GEN + 7 300: the sample at or before GEN + 100 is the submission: produced 0
    assert!(m.stalled(GEN + 7_300));
    let mut m = StallModel::genesis(GEN, w(1_000_000));
    m.submit(GEN + 100, w(1_000_000 + 13_999), w(1_000));
    assert!(m.stalled(GEN + 7_200), "13.999 blocks of work in 2 h: stalled");
    // more than 64 samples: the oldest kept one stands for older times (stall declared earlier)
    // (tipBlockWork 5 000: stalled below 70 000 produced)
    let mut m = StallModel::genesis(GEN, w(0));
    for i in 1..=100u64 {
        m.submit(GEN + 10 * i, w(1_000 * i), w(5_000));
    }
    // the ring keeps submissions 37..=100 (GEN + 370 ..): at GEN + 7 569, no kept sample is at or before
    // GEN + 369, the oldest kept (37 000) stands for it: produced 63 000 < 70 000 (with the genesis
    // sample it would be 100 000)
    assert!(m.stalled(GEN + 7_569));
    // at GEN + 8 199: the sample at or before GEN + 999 is submission 99: produced 1 000
    assert!(m.stalled(GEN + 8_199));
    // at GEN + 7 570: the sample of GEN + 370 itself (37 000): the same 63 000
    assert!(m.stalled(GEN + 7_570));
    // the same submissions, other tip block works: 63 000 produced at GEN + 7 569 is not a stall at 4 000
    // (threshold 56 000; the ring index must not wrap to the newest sample), and is one at 4 540 (threshold
    // 63 560; a ring of more than 64 would keep submission 36 and see 64 000)
    for (tbw, stalled) in [(4_000u64, false), (4_540, true)] {
        let mut m = StallModel::genesis(GEN, w(0));
        for i in 1..=100u64 {
            m.submit(GEN + 10 * i, w(1_000 * i), w(tbw));
        }
        assert_eq!(m.stalled(GEN + 7_569), stalled, "tipBlockWork {tbw}");
    }
    // within W_STALL of the genesis, never stalled, even with 1 block of work in that time
    let mut m = StallModel::genesis(GEN, w(1_000_000));
    m.submit(GEN + 100, w(1_000_000 + 1_000), w(1_000));
    assert!(!m.stalled(GEN + 7_000));
}

/// The relayer of the simulation: a point when due (cadence after the last accepted one), included after
/// `delay` (poll, proof, L1 inclusion) and carrying the chain's work at its proof start.
fn simulate_cadence(cadence: u64, blocks: &[u64], delays: &mut dyn FnMut() -> (u64, u64), until: u64, step: u64) -> (u64, u64) {
    const UNIT: u64 = 1_000;
    let work_at = |t: u64| -> u64 { 1_000_000 + UNIT * blocks.partition_point(|b| *b <= t) as u64 };
    let mut m = StallModel::genesis(GEN, w(work_at(GEN)));
    let mut last = GEN;
    let mut next_due = GEN + cadence;
    let (mut stalled, mut samples) = (0u64, 0u64);
    let (mut d, mut proof) = delays();
    let mut t = GEN;
    while t <= until {
        if t >= next_due + d {
            // the point lands at t; its proof started `proof` seconds before
            m.submit(t, w(work_at(t - proof)), w(UNIT));
            last = t;
            next_due = last + cadence;
            (d, proof) = delays();
        }
        if m.stalled(t) {
            stalled += 1;
        }
        samples += 1;
        t += step;
    }
    let _ = last;
    (stalled, samples)
}

/// xorshift64*: a fixed, reproducible stream.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Exponential block intervals of mean 180 s (a steady hashrate, the target at equilibrium).
fn poisson_blocks(seed: u64, until: u64) -> Vec<u64> {
    let mut r = Rng(seed);
    let mut t = GEN as f64;
    let mut out = Vec::new();
    while (t as u64) < until {
        t += -180.0 * (1.0 - r.unit()).ln();
        out.push(t as u64);
    }
    out
}

/// D-AV3: with one point every 6 600 s (≈ 1 h 50) and any delay below the 600 s margin, `stalled()` is
/// never true in a normal regime — blocks of 180 s on average, here 30 days of exponential intervals over
/// 8 seeds and also blocks every 180 s exactly — sampled every 60 s.
#[test]
fn stalled_is_never_true_in_a_normal_regime_at_the_default_cadence() {
    let until = GEN + 30 * 86_400;
    let exact: Vec<u64> = (1..=(30 * 86_400 / 180)).map(|i| GEN + 180 * i).collect();
    let mut worst = || (CADENCE_MARGIN_SECS - 1, CADENCE_MARGIN_SECS - 1);
    let (s, n) = simulate_cadence(DEFAULT_CHECKPOINT_CADENCE, &exact, &mut worst, until, 60);
    assert_eq!((s, n), (0, 30 * 86_400 / 60 + 1), "exact blocks, the longest delay of the margin");
    for seed in 1..=8u64 {
        let blocks = poisson_blocks(seed, until);
        let mut r = Rng(seed * 7_919);
        let mut delays = || {
            let d = r.next() % CADENCE_MARGIN_SECS;
            (d, r.next() % (d + 1))
        };
        let (s, _) = simulate_cadence(DEFAULT_CHECKPOINT_CADENCE, &blocks, &mut delays, until, 60);
        assert_eq!(s, 0, "seed {seed}: stalled() true on {s} samples");
    }
}

/// The model gives the contract's measured figures (`ADVANCER_INTERFACE.md` §11.1): one point every 6 h ⇒
/// stalled 192 / 288 samples of a day (66.7 %); every 2 h without any delay ⇒ 0 / 288; and a point every
/// W_STALL with any delay stalls: the margin is needed. The same four scenarios run on the real contract
/// (forge) give the same counts.
#[test]
fn the_stall_model_reproduces_the_contracts_measures() {
    let until = GEN + 3 * 86_400;
    let exact: Vec<u64> = (1..=(3 * 86_400 / 180)).map(|i| GEN + 180 * i).collect();
    let day = |cadence: u64, delay: u64| {
        let mut d = || (delay, 0);
        let all = simulate_cadence(cadence, &exact, &mut d, GEN + 2 * 86_400 - 300, 300);
        let first = simulate_cadence(cadence, &exact, &mut d, GEN + 86_400 - 300, 300);
        (all.0 - first.0, all.1 - first.1)
    };
    assert_eq!(day(21_600, 0), (192, 288));
    assert_eq!(day(7_200, 0), (0, 288));
    // the same scenario on the real contract (forge, scratch copy of v3-advance, JGFILL report): 12 / 288
    assert_eq!(day(7_200, 60), (12, 288), "a point every W_STALL with 60 s of delay stalls");
    assert_eq!(day(DEFAULT_CHECKPOINT_CADENCE, CADENCE_MARGIN_SECS - 1), (0, 288));
    let _ = until;
}
