// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! `jg-test-sleep` — **TEST ONLY**: a process that ignores its arguments and
//! waits for SIGTERM, to exercise `ops/jg-start` and `ops/jg-stop`.

fn main() {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}
