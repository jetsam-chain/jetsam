use super::*;
use serde_json::json;

#[test]
fn feed_lines_verify_only_with_their_key_and_body() {
    let k1 = FeedKey::from_secret([1u8; 32]);
    let k2 = FeedKey::from_secret([2u8; 32]);
    let keys: HashMap<u32, [u8; 32]> = [(1, k1.public()), (2, k2.public())].into();
    let line = sign_line(&k1, &json!({"g": 1, "t": 5, "type": "hb"}));
    let (g, body) = verify_line(&line, &keys).unwrap();
    assert_eq!(g, 1);
    assert_eq!(body["type"], "hb");
    // signed by key 2 but claiming g1
    let forged = sign_line(&k2, &json!({"g": 1, "t": 5, "type": "hb"}));
    assert!(verify_line(&forged, &keys).is_err());
    // altered body
    let altered = line.replace("\\\"t\\\":5", "\\\"t\\\":6");
    assert_ne!(altered, line);
    assert!(verify_line(&altered, &keys).is_err());
    // unknown guardian
    let k9 = FeedKey::from_secret([9u8; 32]);
    assert!(verify_line(&sign_line(&k9, &json!({"g": 9})), &keys).is_err());
}

#[test]
fn net_publishes_no_host_identity() {
    let k = FeedKey::from_secret([1u8; 32]);
    let line = sign_line(&k, &json!({"g": 1, "t": 1_800_000_000u64, "type": "hb", "state": "OK"}));
    let host = std::fs::read_to_string("/etc/hostname").unwrap_or_default();
    let host = host.trim();
    if !host.is_empty() {
        assert!(!line.contains(host));
    }
    for forbidden in ["127.0.0.1", "10.77.", "hostname", "tz", "+01"] {
        assert!(!line.contains(forbidden), "{forbidden}");
    }
    assert_eq!(utc_day(1_800_000_000), "2027-01-15");
    assert_eq!(utc_day(0), "1970-01-01");
}

#[test]
fn net_uploads_only_to_configured_destinations() {
    let root = tempfile::tempdir().unwrap();
    let a = root.path().join("pubA");
    let b = root.path().join("pubB");
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    let k = FeedKey::from_secret([1u8; 32]);
    let line = sign_line(&k, &json!({"g": 3, "t": 1}));
    assert_eq!(publish(&[a.clone(), b.clone()], 3, "2027-01-15", &line), 2);
    let entries: Vec<_> = walk(root.path());
    assert_eq!(entries.len(), 2);
    for e in &entries {
        assert!(e.ends_with("feeds/g3/2027-01-15.jsonl"), "{}", e.display());
    }
    // a missing point is not created: publication there fails, the other still counts
    let missing = root.path().join("absent");
    assert_eq!(publish(&[missing.clone(), a.clone()], 3, "2027-01-15", &line), 1);
    assert!(!missing.exists());
}

fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            out.extend(walk(&p));
        } else {
            out.push(p);
        }
    }
    out
}
