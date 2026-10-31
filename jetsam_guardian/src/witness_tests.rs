use super::*;

const NOW: u64 = 1_800_000_000;

fn src(name: &str, tip: u64, h: u64, id: u8, age: u64) -> SourceView {
    let mut ids = BTreeMap::new();
    ids.insert(h, [id; 32]);
    SourceView { name: name.into(), at_unix: NOW - age, tip, ids, authenticated: true }
}

#[test]
fn witness_requires_two_agreeing_sources() {
    let r = WitnessRule::default();
    assert_eq!(witness(100, &[1u8; 32], &[src("g2", 110, 100, 1, 5)], NOW, r), (false, "TOO_FEW_SOURCES".into()));
    assert_eq!(witness(100, &[1u8; 32], &[], NOW, r), (false, "TOO_FEW_SOURCES".into()));
    let two = [src("g2", 110, 100, 1, 5), src("seed1", 109, 100, 1, 1)];
    assert_eq!(witness(100, &[1u8; 32], &two, NOW, r), (true, "OK".into()));
}

#[test]
fn witness_any_disagreeing_source_refuses() {
    let r = WitnessRule::default();
    let three = [src("g2", 110, 100, 1, 5), src("g3", 110, 100, 1, 5), src("seed1", 109, 100, 2, 1)];
    assert_eq!(witness(100, &[1u8; 32], &three, NOW, r), (false, "SOURCES_DISAGREE".into()));
}

#[test]
fn witness_ignores_stale_or_unsigned_feed() {
    let r = WitnessRule::default();
    let mut unsigned = src("g4", 110, 100, 1, 5);
    unsigned.authenticated = false;
    let stale = src("g5", 110, 100, 1, 301);
    let one = src("g2", 110, 100, 1, 300);
    assert_eq!(witness(100, &[1u8; 32], &[one.clone(), unsigned, stale], NOW, r), (false, "TOO_FEW_SOURCES".into()));
    // a stale or unsigned disagreeing source does not veto either
    let mut bad = src("g6", 110, 100, 2, 1);
    bad.authenticated = false;
    let two = [one, src("g3", 110, 100, 1, 1), bad, src("g7", 110, 100, 2, 999)];
    assert_eq!(witness(100, &[1u8; 32], &two, NOW, r), (true, "OK".into()));
}

#[test]
fn witness_source_must_report_height_as_final() {
    let r = WitnessRule::default();
    let shallow = [src("g2", 107, 100, 1, 5), src("g3", 108, 100, 1, 5)];
    assert_eq!(witness(100, &[1u8; 32], &shallow, NOW, r), (false, "TOO_FEW_SOURCES".into()));
    // a source that does not know the height neither agrees nor disagrees
    let silent = SourceView { name: "g4".into(), at_unix: NOW, tip: 200, ids: BTreeMap::new(), authenticated: true };
    let ok = [src("g2", 110, 100, 1, 5), src("g3", 108, 100, 1, 5), silent];
    assert_eq!(witness(100, &[1u8; 32], &ok, NOW, r), (true, "OK".into()));
}
