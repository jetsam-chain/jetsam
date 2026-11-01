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
