use super::*;

const TEXT: &str = r#"
index = 3
generation = 1
network = "d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85"
l1_chain_id = 31337
l1_bridge = "0xee8b5fca072382781999798cf9d7c5b3f6c63e2f"
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

/// The private rehearsal's egress (its seeds and L1 sit on its private bridge 10.77.0.0/24).
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
    assert_eq!(GuardianConfig::parse(&on_mainnet(&onion_sources(TEXT))).unwrap().profile().unwrap(), crate::Profile::Mainnet);
    assert_eq!(GuardianConfig::parse(&onion_sources(TEXT)).unwrap().profile().unwrap(), crate::Profile::PrivateRehearsal);
}

#[test]
fn config_refuses_stability_below_1200_on_mainnet() {
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

// ---- D66: the public test network has the mainnet's rules; the private rehearsal is told apart by its L1 domain ----

const REHEARSAL_DOMAIN: &str = "l1_chain_id = 31337\nl1_bridge = \"0xee8b5fca072382781999798cf9d7c5b3f6c63e2f\"";

/// The fixture as a guardian of the public test network: same genesis (the test chain's), its L1 is Sepolia.
fn on_public_testnet(text: &str) -> String {
    let out = text.replace(REHEARSAL_DOMAIN, "l1_chain_id = 11155111\nl1_bridge = \"0x1111111111111111111111111111111111111111\"");
    assert_ne!(out, text);
    out
}

fn with_stability(text: &str, secs: u64) -> String {
    let out = text.replace("min_dispatchable_peers = 6\n", &format!("min_dispatchable_peers = 6\nstability_secs = {secs}\n"));
    assert_ne!(out, text);
    out
}

fn domain(network: [u8; 32], l1_chain_id: u64, l1_bridge: [u8; 20]) -> crate::Domain {
    crate::Domain { network, l1_chain_id, l1_bridge }
}

#[test]
fn profile_tells_the_three_networks_apart_by_genesis_and_l1_domain() {
    use crate::Profile::*;
    use crate::{MAINNET_NETWORK as MAIN, TESTNET_NETWORK as TEST};
    let bridge = jetsam_chain::consensus::destination::state::REHEARSAL_L1_BRIDGE.address;
    assert_eq!(hex::encode(bridge), "ee8b5fca072382781999798cf9d7c5b3f6c63e2f", "the rehearsal's synthetic bridge");
    let other = [0x5a; 32];
    for (d, p) in [
        (domain(MAIN, 1, [0x11; 20]), Mainnet),
        // the mainnet genesis is the mainnet, whatever L1 it names
        (domain(MAIN, 31_337, bridge), Mainnet),
        (domain(TEST, 11_155_111, [0x11; 20]), PublicTestnet),
        (domain(TEST, 11_155_111, bridge), PublicTestnet),
        // the test chain's genesis is shared with the rehearsal: only the rehearsal's WHOLE L1 domain is the rehearsal
        (domain(TEST, 31_337, [0xee; 20]), PublicTestnet),
        (domain(TEST, 31_338, bridge), PublicTestnet),
        (domain(TEST, 31_337, bridge), PrivateRehearsal),
        // another network: the rehearsal's L1 only; any other L1 is treated as public
        (domain(other, 31_337, [0xee; 20]), PrivateRehearsal),
        (domain(other, 11_155_111, [0xee; 20]), OtherNetwork),
        (domain(other, 1, bridge), OtherNetwork),
    ] {
        assert_eq!(crate::Profile::of(&d), p, "{d:?}");
        assert_eq!(p.strict(), p != PrivateRehearsal);
    }
    assert!(Mainnet.strict() && PublicTestnet.strict() && OtherNetwork.strict() && !PrivateRehearsal.strict());
    assert_eq!(crate::REHEARSAL_L1_CHAIN_ID, 31_337);
}

#[test]
fn testnet_network_is_the_test_chain_genesis_id() {
    use jetsam_chain::consensus::genesis::{genesis_header, GENESIS_TIMESTAMP};
    let id = jetsam_chain::block_header::block_id(&genesis_header());
    assert_eq!(hex::encode(crate::TESTNET_NETWORK), "d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85");
    assert_ne!(crate::TESTNET_NETWORK, crate::MAINNET_NETWORK);
    if GENESIS_TIMESTAMP == 1_791_385_200 {
        assert_eq!(id, crate::TESTNET_NETWORK, "a test-chain build: the test chain's genesis");
    } else {
        assert_ne!(id, crate::TESTNET_NETWORK, "the mainnet build");
    }
}

#[test]
fn config_clearnet_egress_is_refused_on_the_public_testnet() {
    let e = err(&on_public_testnet(&format!("{TEXT}{CLEARNET}")));
    assert!(e.contains("clearnet") && e.contains("public test network") && e.contains("does not start"), "{e}");
    // even with sources that would be fine in onion mode
    assert!(GuardianConfig::parse(&on_public_testnet(&format!("{}{CLEARNET}", onion_sources(TEXT)))).is_err());
    let c = GuardianConfig::parse(&on_public_testnet(&onion_sources(TEXT))).unwrap();
    assert_eq!(c.profile().unwrap(), crate::Profile::PublicTestnet);
    // the test chain's genesis with the rehearsal's chain id but another bridge is not the rehearsal
    let half = TEXT.replace("0xee8b5fca072382781999798cf9d7c5b3f6c63e2f", "0xee8b5fca00000000000000000000000000003e2f");
    let e = err(&format!("{half}{CLEARNET}"));
    assert!(e.contains("clearnet") && e.contains("public test network"), "{e}");
    // an unknown network that names a public L1 is not the rehearsal either
    let other = on_public_testnet(&TEXT.replace("d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85", &"5a".repeat(32)));
    assert!(err(&format!("{other}{CLEARNET}")).contains("clearnet"));
    // the private rehearsal keeps its clearnet
    assert_eq!(GuardianConfig::parse(&format!("{TEXT}{CLEARNET}")).unwrap().egress().unwrap(), crate::rpc::Egress::PrivateBridge);
}

#[test]
fn config_rehearsal_clearnet_reaches_only_the_loopback_and_its_bridge() {
    let seeds = |list: &str| format!("{}{CLEARNET}", TEXT.replace(REHEARSAL_SEED, &format!("seed_rpcs = [{list}]")));
    let ok = seeds(r#""http://10.77.0.1:19711", "http://10.77.0.254:19721", "http://127.0.0.1:19731", "http://[::1]:19741""#);
    assert!(GuardianConfig::parse(&ok).is_ok(), "{:?}", GuardianConfig::parse(&ok).err());
    let l1 = format!("{ok}").replace("seed_rpcs = [", "l1_rpc = \"http://10.77.0.1:18545\"\nseed_rpcs = [");
    assert!(GuardianConfig::parse(&l1).is_ok());
    for bad in [
        "http://10.77.1.1:19711",
        "http://10.78.0.1:19711",
        "http://10.76.0.1:19711",
        "http://192.168.1.10:19711",
        "http://203.0.113.7:19711",
        "http://[::ffff:10.77.0.1]:19711",
        "http://seed.example.org:19711",
        "http://localhost:19711",
        "http://mseed1.jetsamchain.com:19711",
    ] {
        let e = err(&seeds(&format!("\"http://10.77.0.1:19711\", \"{bad}\"")));
        assert!(e.contains("seed_rpcs[1]") && e.contains("does not start"), "{bad}: {e}");
        let e = err(&l1.replace("http://10.77.0.1:18545", bad));
        assert!(e.contains("l1_rpc") && e.contains("does not start"), "{bad}: {e}");
    }
    let e = err(&seeds(&format!("\"http://{}:19711\"", onion('a'))));
    assert!(e.contains("seed_rpcs[0]") && e.contains(".onion"), "{e}");
}

#[test]
fn config_refuses_stability_below_1200_on_the_public_testnet() {
    let public = on_public_testnet(&onion_sources(TEXT));
    let e = err(&with_stability(&public, 1_199));
    assert!(e.contains("stability_secs") && e.contains("1200") && e.contains("public test network"), "{e}");
    assert!(GuardianConfig::parse(&with_stability(&public, 0)).is_err());
    assert_eq!(GuardianConfig::parse(&with_stability(&public, 1_200)).unwrap().thresholds.stability_secs, 1_200);
    assert_eq!(GuardianConfig::parse(&public).unwrap().thresholds.stability_secs, 1_200, "the default is unchanged");
    let other = public.replace("d9d156ee35e725cf6b133216809c1b0f165d1120c9cf42cc2bfeb0ef9031ee85", &"5a".repeat(32));
    assert!(GuardianConfig::parse(&with_stability(&other, 600)).is_err(), "an unknown network naming a public L1");
    // the private rehearsal keeps its own values, in both modes
    assert_eq!(GuardianConfig::parse(&with_stability(&format!("{TEXT}{CLEARNET}"), 600)).unwrap().thresholds.stability_secs, 600);
}

#[test]
fn config_node_rpc_must_be_loopback_or_onion_in_every_profile() {
    let node = |text: &str, url: &str| {
        let out = text.replace("node_rpc = \"http://127.0.0.1:19801\"", &format!("node_rpc = \"{url}\""));
        assert_ne!(out, text);
        out
    };
    let rehearsal = format!("{TEXT}{CLEARNET}");
    for base in [onion_sources(TEXT), on_public_testnet(&onion_sources(TEXT)), on_mainnet(&onion_sources(TEXT)), rehearsal.clone()] {
        for bad in [
            "http://10.77.0.1:19801",
            "http://192.168.1.2:19801",
            "http://localhost:19801",
            "http://node.example.org:19801",
            "https://127.0.0.1:19801",
            "http://[::ffff:127.0.0.1]:19801",
        ] {
            let e = err(&node(&base, bad));
            assert!(e.contains("node_rpc") && e.contains("does not start"), "{bad}: {e}");
        }
        assert!(GuardianConfig::parse(&node(&base, "http://[::1]:19801")).is_ok());
        let c = GuardianConfig::parse(&node(&base, &format!("http://{}:19801", onion('n')))).unwrap();
        assert_eq!(c.node_egress().unwrap(), crate::rpc::Egress::Onion { socks: "127.0.0.1:9050".parse().unwrap() });
    }
    // the configured SOCKS port is the one used for an onion node
    let text = format!("{}\n[egress]\ntor_socks = \"127.0.0.1:9150\"\n", onion_sources(TEXT));
    assert_eq!(GuardianConfig::parse(&text).unwrap().node_egress().unwrap(), crate::rpc::Egress::Onion { socks: "127.0.0.1:9150".parse().unwrap() });
    // the rehearsal's own node is on the loopback as well
    assert_eq!(GuardianConfig::parse(&rehearsal).unwrap().node_egress().unwrap(), crate::rpc::Egress::Onion { socks: "127.0.0.1:9050".parse().unwrap() });
}

#[test]
fn checked_egress_without_a_domain_refuses_clearnet() {
    // jg-aggregate's case: a program that may not know its domain
    let clearnet = EgressConfig { mode: EgressMode::Clearnet, tor_socks: "127.0.0.1:9050".into() };
    let l1 = || [("l1_rpc".to_string(), "http://10.77.0.1:18545")];
    let e = format!("{:#}", checked_egress(&clearnet, None, l1()).unwrap_err());
    assert!(e.contains("clearnet") && e.contains("domain"), "{e}");
    let bridge = jetsam_chain::consensus::destination::state::REHEARSAL_L1_BRIDGE.address;
    let rehearsal = domain(crate::TESTNET_NETWORK, 31_337, bridge);
    assert_eq!(checked_egress(&clearnet, Some(&rehearsal), l1()).unwrap(), crate::rpc::Egress::PrivateBridge);
    let public = domain(crate::TESTNET_NETWORK, 11_155_111, bridge);
    assert!(format!("{:#}", checked_egress(&clearnet, Some(&public), l1()).unwrap_err()).contains("public test network"));
    let onion_cfg = EgressConfig::default();
    assert!(checked_egress(&onion_cfg, None, l1()).is_err(), "onion mode: the bridge is refused");
    let ok = format!("http://{}:8545", onion('l'));
    assert_eq!(
        checked_egress(&onion_cfg, None, [("l1_rpc".to_string(), ok.as_str()), ("robot_rpc".to_string(), "http://127.0.0.1:19751")]).unwrap(),
        crate::rpc::Egress::Onion { socks: "127.0.0.1:9050".parse().unwrap() }
    );
}

// ---- C-2 (a) (audit 2026-11-10): the feeds pulled over Tor, never pushed ----

/// The mainnet fixture (guardian g3, feed keys g1 and g2) with `feed_listen` and `[feed_onions]`.
fn feeds_on_mainnet(listen: Option<&str>, onions: &[(&str, &str)]) -> String {
    let listen = listen.map(|l| format!("feed_listen = \"{l}\"\n")).unwrap_or_default();
    let text = on_mainnet(&onion_sources(TEXT))
        .replace("\n[feed_keys]\n", &format!("\n{listen}[feed_keys]\ng2 = \"{}\"\n", "22".repeat(32)));
    let onions: String = onions.iter().map(|(g, url)| format!("{g} = \"{url}\"\n")).collect();
    if onions.is_empty() { text } else { format!("{text}\n[feed_onions]\n{onions}") }
}

/// On a public network a guardian serves its feed read-only on the loopback (behind its onion service) and names the
/// onion server of every other guardian; without them it does not start. The private rehearsal may keep directories.
#[test]
fn c2a_a_public_guardian_serves_its_feed_and_pulls_every_other_over_tor() {
    let (a, b) = (format!("http://{}.onion:8080", "a".repeat(56)), format!("http://{}.onion", "b".repeat(56)));
    let refused = |text: &str| GuardianConfig::parse(text).unwrap().feed_transport().unwrap_err().to_string();
    let e = refused(&feeds_on_mainnet(None, &[("g1", &a), ("g2", &b)]));
    assert!(e.contains("feed_listen") && e.contains("does not start"), "{e}");
    let e = refused(&feeds_on_mainnet(Some("127.0.0.1:8080"), &[("g2", &b)]));
    assert!(e.contains("g1") && e.contains("does not start"), "{e}");
    let ok = GuardianConfig::parse(&feeds_on_mainnet(Some("127.0.0.1:8080"), &[("g1", &a), ("g2", &b)])).unwrap();
    let (listen, peers) = ok.feed_transport().unwrap();
    assert_eq!(listen, Some("127.0.0.1:8080".parse().unwrap()));
    assert_eq!(peers, vec![(1, a.clone()), (2, b.clone())]);
    for (listen, onions, why) in [
        ("0.0.0.0:8080", vec![("g1", a.as_str()), ("g2", b.as_str())], "loopback"),
        ("127.0.0.1:8080", vec![("g1", a.as_str()), ("g2", "http://feeds.example:80")], "g2"),
        ("127.0.0.1:8080", vec![("g1", a.as_str()), ("g2", "http://127.0.0.1:8081")], ".onion"),
        ("127.0.0.1:8080", vec![("g1", a.as_str()), ("g2", b.as_str()), ("g3", a.as_str())], "g3"),
        ("127.0.0.1:8080", vec![("g1", a.as_str()), ("g2", b.as_str()), ("g9", a.as_str())], "g9"),
    ] {
        let e = refused(&feeds_on_mainnet(Some(listen), &onions));
        assert!(e.contains(why) && e.contains("does not start"), "{listen} {onions:?}: {e}");
    }
    // the rehearsal: directories alone still start
    assert_eq!(GuardianConfig::parse(&onion_sources(TEXT)).unwrap().feed_transport().unwrap(), (None, vec![]));
}

/// AN-1 (audit n°2): the egress of the return-over-Base robots is `jg-relay`'s. The command line deployed on the test
/// chain (`fast-run --l1 http://127.0.0.1:8560 --base …:8548 --base …:8545 --beacon …:8551 --node …:9711`), with no new
/// flag, starts: onion by default, every source on the loopback reached directly (never through Tor). A public address
/// in clear (an IP address or a name over http://) is refused before any socket; an https:// public name goes through
/// Tor; `--tor-socks` must be a loopback address; `--egress` is onion or clearnet, and clearnet is refused on a public
/// network (this build's).
#[test]
fn an1_robot_egress_reaches_the_loopback_directly_and_refuses_a_public_address_in_clear() {
    use crate::rpc::{Egress, Route};
    let line = |extra: &[&str]| -> Vec<String> {
        [
            "--key", "relay.secret", "--l1", "http://127.0.0.1:8560", "--base", "http://127.0.0.1:8548", "--base",
            "http://127.0.0.1:8545", "--beacon", "http://127.0.0.1:8551", "--node", "http://127.0.0.1:9711", "--node-cookie",
            "rpc.cookie", "--state-dir", "fast", "--from-block", "57071077", "--interval", "15",
        ]
        .iter()
        .chain(extra)
        .map(|a| a.to_string())
        .collect()
    };
    let deployed = line(&[]);
    let sources = crate::fast_run::robot_sources(&deployed, &["--l1", "--base", "--beacon", "--node"]);
    assert_eq!(sources.len(), 5, "{sources:?}");
    let egress = robot_egress(&deployed, sources.clone()).expect("the deployed line starts");
    assert_eq!(egress, Egress::Onion { socks: "127.0.0.1:9050".parse().unwrap() });
    for (name, url) in &sources {
        assert!(matches!(egress.route(url).unwrap(), Route::Loopback(_)), "{name} {url}: the loopback directly");
    }
    for public in ["http://203.0.113.5:8545", "http://ethereum-sepolia.example.org", "http://[2001:db8::1]:8545"] {
        let e = format!("{:#}", robot_egress(&deployed, [("--l1".to_string(), public)]).unwrap_err());
        assert!(e.contains("onion-only egress refuses") && e.contains(public), "{public}: {e}");
    }
    let https = "https://ethereum-sepolia.example.org/v1";
    let egress = robot_egress(&deployed, [("--l1".to_string(), https)]).unwrap();
    assert!(matches!(egress.route(https).unwrap(), Route::Socks { .. }), "an https:// public name through Tor");
    let other = robot_egress(&line(&["--tor-socks", "127.0.0.1:9150"]), []).unwrap();
    assert_eq!(other, Egress::Onion { socks: "127.0.0.1:9150".parse().unwrap() });
    assert!(robot_egress(&line(&["--tor-socks", "10.0.0.1:9050"]), []).is_err(), "a SOCKS port off the machine");
    assert!(robot_egress(&line(&["--tor-socks"]), []).is_err());
    let e = format!("{:#}", robot_egress(&line(&["--egress", "tor"]), []).unwrap_err());
    assert!(e.contains("onion or clearnet"), "{e}");
    assert_eq!(robot_egress(&line(&["--egress", "onion"]), sources.clone()).unwrap(), egress_default());
    assert!(crate::Profile::of(&robot_domain()).strict(), "this build is a public network's");
    let e = format!("{:#}", robot_egress(&line(&["--egress", "clearnet"]), sources).unwrap_err());
    assert!(e.contains("clearnet") && e.contains("refused on"), "{e}");
}

fn egress_default() -> crate::rpc::Egress {
    crate::rpc::Egress::onion(crate::rpc::DEFAULT_TOR_SOCKS).unwrap()
}
