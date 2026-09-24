// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Why the two lines a miner watches carry a symbol and not colour.
//!
//! A mining node prints a great deal that matters to the node and nothing to
//! the person running it. Two lines matter to that person: how fast the
//! machine is searching, and whether it won a block. Both used to arrive in
//! the same shape as the sync chatter scrolling past them, and the first
//! operator to run this reported seeing neither — while the log in front of
//! them held three rate lines and a won block.
//!
//! # Colour is not available here, and that is deliberate upstream
//!
//! `tracing` escapes control characters in anything the caller supplies, for
//! both the message body and structured fields:
//!
//! ```text
//! tracing::info!("{}", "\x1b[1;36mX\x1b[0m")  ->  message: \x1b[1;36mX\x1b[0m
//! tracing::info!(f = "\x1b[1;36mX\x1b[0m")     ->  f="\u{1b}[1;36mX\u{1b}[0m"
//! ```
//!
//! That is a defence against escape-sequence injection through log content,
//! and it applies to us exactly as it applies to a hostile peer name. The
//! subscriber colours what it owns — the level, the target — and nothing an
//! event carries. An earlier attempt at colouring these two lines shipped in
//! v1.4.1 and printed the escape codes as literal text on every terminal.
//!
//! # What is used instead
//!
//! A leading symbol, which is an ordinary UTF-8 character and survives
//! untouched, plus word order: the thing worth reading comes first. This also
//! behaves identically in a terminal, in a file and under `journalctl`, where
//! colour never applied anyway.

/// Marks a measured search rate.
pub const MINING: &str = "⛏";

/// Marks a block this node won.
pub const WON: &str = "✅";

#[cfg(test)]
mod tests {
    /// No escape byte may ever reach a `tracing` event from this crate.
    ///
    /// The earlier version of this module wrapped strings in ANSI codes and
    /// was proven only against a redirected sink, where the wrapping was
    /// disabled — so the test passed on the one path that could not fail. The
    /// property that matters is unconditional: never hand `tracing` a control
    /// character, because it will escape it and print it as text.
    #[test]
    fn the_markers_carry_no_control_characters() {
        for marker in [super::MINING, super::WON] {
            assert!(
                !marker.chars().any(char::is_control),
                "a control character would be printed literally by tracing: {marker:?}"
            );
        }
    }
}
