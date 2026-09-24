// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Colour for the two lines a miner actually watches.
//!
//! A mining node prints a great deal that matters to the node and nothing to
//! the person running it. Two lines matter to the person: how fast the machine
//! is searching, and whether it won a block. Both used to arrive in the same
//! grey as the snapshot chatter scrolling past them, and operators reported
//! seeing neither.
//!
//! # Rules
//!
//! * **A terminal, never a file.** Escape sequences written to a redirected
//!   log make `grep` match on invisible bytes and turn a log into noise
//!   wherever it is later read. The node's subscriber already decides this the
//!   same way; this decides it once, for message bodies, which the subscriber
//!   does not touch.
//! * **`NO_COLOR` is honoured**, as an environment variable of any value —
//!   the convention at <https://no-color.org>.
//! * **The keywords stay outside the colour.** `block accepted` and `kH/s`
//!   remain plain text at a stable position, because scripts grep for them and
//!   a colour code between the words would break every one of them.

use std::io::IsTerminal;
use std::sync::OnceLock;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
    })
}

fn wrap(code: &str, s: &str) -> String {
    if enabled() {
        format!("\x1b[{code}m{s}\x1b[0m")
    } else {
        s.to_string()
    }
}

/// A measured rate: the number an operator checks to know the machine works.
pub fn rate(s: &str) -> String {
    wrap("1;36", s) // bold cyan
}

/// A block this node won. The one line worth scrolling back for.
pub fn won(s: &str) -> String {
    wrap("1;32", s) // bold green
}

/// Context that should stay readable without competing with the two above.
pub fn faint(s: &str) -> String {
    wrap("2", s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing may be wrapped when the sink is not a terminal.
    ///
    /// Tests capture stdout, so `enabled()` is false here and every helper is
    /// the identity. That is the property worth pinning: a redirected log — a
    /// file, a pipe, a systemd journal — must carry no escape byte at all.
    #[test]
    fn a_redirected_sink_gets_no_escape_bytes() {
        for painted in [rate("8.30 kH/s"), won("block accepted"), faint("h=5629")] {
            assert!(
                !painted.contains('\x1b'),
                "escape sequence reached a non-terminal sink: {painted:?}"
            );
        }
    }

    /// The words scripts grep for must survive colouring untouched.
    #[test]
    fn keywords_are_never_split_by_a_colour_code() {
        assert!(rate("8.30 kH/s").contains("kH/s"));
        assert!(won("block accepted").contains("block accepted"));
    }
}
