// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Guardians in consensus (T1.1 design, `jetsam-v17-gardiens-consensus_DESIGN.md`):
//! the tests of P3, one file per task.

use serde_json::Value;

mod constants;

/// A fixture of `guardian/testdata/`.
pub(crate) fn fixture(name: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/consensus/guardian/testdata")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture")).expect("json")
}

/// The bytes of a `0x…` hex string.
pub(crate) fn unhex(text: &str) -> Vec<u8> {
    let text = text.strip_prefix("0x").unwrap_or(text);
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}
