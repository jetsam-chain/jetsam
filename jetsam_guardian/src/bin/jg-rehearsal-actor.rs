// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-rehearsal-actor <name>` — **TEST ONLY**: the spend secret (hex) of the
//! rehearsal actor `name`, the key `jetsam-rehearsal` derives from a public
//! label (`poseidon2b("JTM/REHEARSAL/V1.7/ACTOR-TEST-KEY", name)`), to hold it
//! in a rehearsal node's wallet (`jetsam --import-spend-secret`). Nothing it
//! prints has value.

fn main() {
    let name = std::env::args().nth(1).expect("jg-rehearsal-actor <name>");
    let secret = jetsam_poseidon2b::native::poseidon2b_hash_bytes(b"JTM/REHEARSAL/V1.7/ACTOR-TEST-KEY", name.as_bytes());
    println!("{}", hex::encode(secret));
}
