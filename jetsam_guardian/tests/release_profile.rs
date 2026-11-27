// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! I-3 (external audit of the robots, decision D-8): the robots' release builds check their arithmetic (an overflow
//! stops the robot instead of wrapping a balance, a cap or a count), the node's do not change.

#[test]
fn d8_the_robots_release_builds_check_overflows_and_the_nodes_are_unchanged() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../Cargo.toml")).unwrap();
    let root: toml::Value = toml::from_str(&text).unwrap();
    let release = &root["profile"]["release"];
    assert_eq!(
        release["package"]["jetsam_guardian"]["overflow-checks"].as_bool(),
        Some(true),
        "[profile.release.package.jetsam_guardian] overflow-checks = true"
    );
    assert!(release.get("overflow-checks").is_none(), "the workspace's release profile (the node's) is not changed");
    let others: Vec<&String> = release.get("package").and_then(|p| p.as_table()).map(|t| t.keys().filter(|k| *k != "jetsam_guardian").collect()).unwrap_or_default();
    assert!(others.is_empty(), "no other package's release build is changed: {others:?}");
}
