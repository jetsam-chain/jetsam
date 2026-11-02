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

/// The private rehearsal's egress (its seeds and L1 sit on a private bridge).
const CLEARNET: &str = "\n[egress]\nmode = \"clearnet\"\n";

#[test]
fn config_parses_and_keeps_design_defaults() {
    // (the fixture's seed is on the private rehearsal bridge: clearnet, said so)
    let text = format!("{TEXT}{CLEARNET}");
    let c = GuardianConfig::parse(&text).unwrap();
    assert_eq!(c.index, 3);
    assert_eq!(c.genesis_go().unwrap(), Some((120, [0x11u8; 32])));
    assert_eq!(c.thresholds.min_dispatchable_peers, 6);
    assert_eq!(c.thresholds.sign_depth, 8, "unset thresholds keep the design's defaults");
    assert_eq!(c.thresholds.stability_secs, 1_200);
    assert_eq!(c.feed_keys().unwrap()[&1], [1u8; 32]);
    assert_eq!(c.domain().unwrap().l1_chain_id, 31_337);
    assert_eq!(c.min_sources, 2);
    assert!(GuardianConfig::parse(&format!("unknown_key = 1\n{text}")).is_err(), "unknown keys refused");
    let bad = text.replace("120:", "x:");
    assert!(GuardianConfig::parse(&bad).unwrap().genesis_go().is_err());
}

// ---- A4-1 / D60: onion-only egress; A4-3: SG-4 floor on mainnet; A4-2: socket uids ----

const REHEARSAL_SEED: &str = r#"seed_rpcs = ["http://10.77.0.1:19711"]"#;

fn onion(c: char) -> String {
    format!("jetsamtest{}.onion", c.to_string().repeat(46))
}

/// The fixture with onion-only sources (a loopback seed, an onion seed, an onion L1).
fn onion_sources(text: &str) -> String {
    let (a, b) = (onion('a'), onion('b'));
    text.replace(REHEARSAL_SEED, &format!("seed_rpcs = [\"http://127.0.0.1:19711\", \"http://{a}:19721\"]\nl1_rpc = \"http://{b}:8545\""))
}

fn on_mainnet(text: &str) -> String {
    text.replace("d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85", &hex::encode(crate::MAINNET_NETWORK))
}

fn err(text: &str) -> String {
    format!("{:#}", GuardianConfig::parse(text).unwrap_err())
}

#[test]
fn config_is_onion_only_by_default_and_refuses_a_clearnet_seed() {
    let e = err(TEXT);
    assert!(e.contains("seed_rpcs[0]") && e.contains("onion-only") && e.contains("does not start"), "{e}");
    let c = GuardianConfig::parse(&onion_sources(TEXT)).unwrap();
    assert_eq!(c.egress, EgressConfig { mode: EgressMode::Onion, tor_socks: "127.0.0.1:9050".into() });
    assert_eq!(c.egress().unwrap(), crate::rpc::Egress::Onion { socks: "127.0.0.1:9050".parse().unwrap() });
    // a second seed in the clear is refused as well
    let two = TEXT.replace(REHEARSAL_SEED, r#"seed_rpcs = ["http://127.0.0.1:19711", "http://mseed1.jetsamchain.com:19711"]"#);
    assert!(err(&two).contains("seed_rpcs[1]"), "{}", err(&two));
}

#[test]
fn config_onion_only_refuses_an_l1_that_is_not_loopback_nor_onion() {
    let ok = onion_sources(TEXT);
    for l1 in ["http://10.77.0.1:18545", "http://localhost:8545", "http://eth.example.org:8545", "https://127.0.0.1:8545"] {
        let text = ok.replace(&format!("l1_rpc = \"http://{}:8545\"", onion('b')), &format!("l1_rpc = \"{l1}\""));
        assert_ne!(text, ok);
        let e = err(&text);
        assert!(e.contains("l1_rpc") && e.contains("does not start"), "{l1}: {e}");
    }
    let local = ok.replace(&format!("l1_rpc = \"http://{}:8545\"", onion('b')), "l1_rpc = \"http://127.0.0.1:8545\"");
    assert!(GuardianConfig::parse(&local).is_ok(), "a local L1 node");
}

#[test]
fn config_tor_socks_must_be_a_loopback_address() {
    let text = format!("{}\n[egress]\ntor_socks = \"10.66.0.1:9050\"\n", onion_sources(TEXT));
    assert!(err(&text).contains("tor_socks"), "{}", err(&text));
    let text = format!("{}\n[egress]\ntor_socks = \"127.0.0.1:9150\"\n", onion_sources(TEXT));
    assert_eq!(GuardianConfig::parse(&text).unwrap().egress().unwrap(), crate::rpc::Egress::Onion { socks: "127.0.0.1:9150".parse().unwrap() });
}

#[test]
fn config_clearnet_egress_is_refused_on_mainnet() {
    let e = err(&on_mainnet(&format!("{TEXT}{CLEARNET}")));
    assert!(e.contains("clearnet") && e.contains("mainnet"), "{e}");
    // even with sources that would be fine in onion mode
    assert!(GuardianConfig::parse(&on_mainnet(&format!("{}{CLEARNET}", onion_sources(TEXT)))).is_err());
    assert!(GuardianConfig::parse(&on_mainnet(&onion_sources(TEXT))).unwrap().is_mainnet());
    assert!(!GuardianConfig::parse(&onion_sources(TEXT)).unwrap().is_mainnet());
}

#[test]
fn config_refuses_stability_below_1200_on_mainnet_only() {
    let with = |text: &str, secs: u64| text.replace("min_dispatchable_peers = 6\n", &format!("min_dispatchable_peers = 6\nstability_secs = {secs}\n"));
    let main = on_mainnet(&onion_sources(TEXT));
    let e = err(&with(&main, 1_199));
    assert!(e.contains("stability_secs") && e.contains("1200"), "{e}");
    assert!(GuardianConfig::parse(&with(&main, 0)).is_err());
    assert_eq!(GuardianConfig::parse(&with(&main, 1_200)).unwrap().thresholds.stability_secs, 1_200);
    assert_eq!(GuardianConfig::parse(&main).unwrap().thresholds.stability_secs, 1_200, "the default is unchanged");
    // the private rehearsal keeps its own values
    assert_eq!(GuardianConfig::parse(&with(&onion_sources(TEXT), 600)).unwrap().thresholds.stability_secs, 600);
}

#[test]
fn config_ipc_uids_default_to_the_server_uid() {
    let c = GuardianConfig::parse(&onion_sources(TEXT)).unwrap();
    assert_eq!(c.ipc, IpcConfig::default());
    assert_eq!((c.ipc.observe_or(1_000), c.ipc.net_or(1_000)), (1_000, 1_000));
    let text = format!("{}\n[ipc]\nobserve_uid = 4243\nnet_uid = 4242\n", onion_sources(TEXT));
    let c = GuardianConfig::parse(&text).unwrap();
    assert_eq!((c.ipc.observe_or(1_000), c.ipc.net_or(1_000)), (4_243, 4_242));
    assert!(GuardianConfig::parse(&format!("{}\n[ipc]\nsign_uid = 1\n", onion_sources(TEXT))).is_err(), "unknown keys refused");
}

#[test]
fn mainnet_network_is_the_mainnet_genesis_id() {
    use jetsam_chain::consensus::genesis::{genesis_header, GENESIS_TIMESTAMP};
    let id = jetsam_chain::block_header::block_id(&genesis_header());
    if GENESIS_TIMESTAMP == 1_787_328_000 {
        assert_eq!(id, crate::MAINNET_NETWORK, "the mainnet genesis");
    } else {
        assert_ne!(id, crate::MAINNET_NETWORK, "a test-chain build");
    }
}
