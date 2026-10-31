use super::*;

const TEXT: &str = r#"
index = 3
generation = 1
network = "d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85"
l1_chain_id = 31337
l1_bridge = "0xee8b5fca00000000000000000000000000003e2f"
v1_7_height = 96
deposits_from_height = 128
genesis_go = "120:1111111111111111111111111111111111111111111111111111111111111111"
node_version = "jetsam/1.6.0 (v1.6=64, v1.7=96)"
attest_every = 16
node_rpc = "http://127.0.0.1:19801"
seed_rpcs = ["http://10.77.0.1:19711"]
publication_points = ["/tmp/a", "/tmp/b"]
group_key = "920e80685f925de21c4e7278bc5377bd2b83a61eed721ea33b8d1606b1b607269f2ec711a61b7e5d75d9c87cd9222e57"
commitments = []
state_dir = "/tmp/g3"

[feed_keys]
g1 = "0101010101010101010101010101010101010101010101010101010101010101"

[thresholds]
min_dispatchable_peers = 6
"#;

#[test]
fn config_parses_and_keeps_design_defaults() {
    let c = GuardianConfig::parse(TEXT).unwrap();
    assert_eq!(c.index, 3);
    assert_eq!(c.genesis_go().unwrap(), Some((120, [0x11u8; 32])));
    assert_eq!(c.thresholds.min_dispatchable_peers, 6);
    assert_eq!(c.thresholds.sign_depth, 8, "unset thresholds keep the design's defaults");
    assert_eq!(c.thresholds.stability_secs, 1_200);
    assert_eq!(c.feed_keys().unwrap()[&1], [1u8; 32]);
    assert_eq!(c.domain().unwrap().l1_chain_id, 31_337);
    assert_eq!(c.min_sources, 2);
    assert!(GuardianConfig::parse(&format!("unknown_key = 1\n{TEXT}")).is_err(), "unknown keys refused");
    let bad = TEXT.replace("120:", "x:");
    assert!(GuardianConfig::parse(&bad).unwrap().genesis_go().is_err());
}
