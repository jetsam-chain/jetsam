// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 the Jetsam developers.

//! Colour for the two lines a miner watches, applied where it survives.
//!
//! A mining node prints a great deal that matters to the node and nothing to
//! the person running it. Two lines matter to that person — the search rate,
//! and a block won — and they used to scroll past in the same shape as the
//! sync traffic around them.
//!
//! # Why the colour is not in the message
//!
//! `tracing` escapes control characters in anything an event carries, message
//! body and fields alike, as a defence against escape-sequence injection
//! through log content:
//!
//! ```text
//! tracing::info!("{}", "\x1b[36mX\x1b[0m")  ->  message: \x1b[36mX\x1b[0m
//! ```
//!
//! An attempt to colour these lines from the miner crate shipped once and
//! printed its own escape codes as literal text on every terminal. The
//! subscriber colours what it owns — level, timestamp, field names — and
//! nothing an event supplies.
//!
//! # Where it goes instead
//!
//! Here: a writer that wraps a whole formatted line once `tracing` is done
//! with it. Nothing is escaped at this point, because nothing inspects it
//! again. The line is recognised by the marker the miner already puts at its
//! front, so the two crates stay decoupled — this file knows two characters,
//! not the miner's message format.
//!
//! Colour goes to a terminal and never to a file: escape bytes in a
//! redirected log make `grep` match on invisible characters. That decision is
//! made once, by the caller, and passed in.

use std::io::{self, Write};

/// Rate lines. Bold cyan: read often, never urgent.
const RATE: &[u8] = "⛏".as_bytes();
const RATE_ON: &[u8] = b"\x1b[1;36m";

/// Won blocks. Bold green: the line worth scrolling back for.
const WON: &[u8] = "✅".as_bytes();
const WON_ON: &[u8] = b"\x1b[1;32m";

const OFF: &[u8] = b"\x1b[0m";

/// Wraps a formatted line in colour when it carries a miner marker.
///
/// Buffers until the newline `tracing` writes at the end of an event, so a
/// line split across several `write` calls is still recognised as one.
pub struct Highlighter<W: Write> {
    inner: W,
    buf: Vec<u8>,
    enabled: bool,
}

impl<W: Write> Highlighter<W> {
    pub fn new(inner: W, enabled: bool) -> Self {
        Self { inner, buf: Vec::with_capacity(256), enabled }
    }

    /// The colour a line earns, or none.
    fn tint(line: &[u8]) -> Option<&'static [u8]> {
        if contains(line, WON) {
            Some(WON_ON)
        } else if contains(line, RATE) {
            Some(RATE_ON)
        } else {
            None
        }
    }

    fn emit(&mut self, line: &[u8]) -> io::Result<()> {
        match self.enabled.then(|| Self::tint(line)).flatten() {
            // Trailing newline stays outside the reset, so a colour never
            // bleeds into whatever the next line is.
            Some(on) => {
                let body = line.strip_suffix(b"\n").unwrap_or(line);
                self.inner.write_all(on)?;
                self.inner.write_all(body)?;
                self.inner.write_all(OFF)?;
                if line.ends_with(b"\n") {
                    self.inner.write_all(b"\n")?;
                }
                Ok(())
            }
            None => self.inner.write_all(line),
        }
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

impl<W: Write> Write for Highlighter<W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.buf.extend_from_slice(data);
        while let Some(end) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=end).collect();
            self.emit(&line)?;
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        if !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            self.emit(&rest)?;
        }
        self.inner.flush()
    }
}

impl<W: Write> Drop for Highlighter<W> {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(lines: &[&str], enabled: bool) -> String {
        let mut sink = Vec::new();
        {
            let mut h = Highlighter::new(&mut sink, enabled);
            for l in lines {
                h.write_all(l.as_bytes()).unwrap();
            }
            h.flush().unwrap();
        }
        String::from_utf8(sink).unwrap()
    }

    #[test]
    fn a_rate_line_is_tinted_and_an_ordinary_one_is_not() {
        let out = run(&["12:00:00 INFO ⛏  8.3 kH/s\n", "12:00:01 INFO peer connected\n"], true);
        assert!(out.contains("\x1b[1;36m"), "the rate line carries no colour: {out:?}");
        assert!(
            !out.lines().nth(1).unwrap().contains('\x1b'),
            "an ordinary line was tinted: {out:?}"
        );
    }

    #[test]
    fn a_won_block_gets_its_own_colour() {
        let out = run(&["12:00:00 INFO ✅ BLOCK WON #5629\n"], true);
        assert!(out.contains("\x1b[1;32m"));
    }

    /// The property the previous attempt got wrong: a redirected sink must
    /// stay free of escape bytes, or `grep` starts matching invisible
    /// characters. Disabled means byte-for-byte identical output.
    #[test]
    fn disabled_is_byte_for_byte_the_input() {
        let lines = ["12:00:00 INFO ⛏  8.3 kH/s\n", "12:00:01 INFO ✅ BLOCK WON #1\n"];
        let out = run(&lines, false);
        assert_eq!(out, lines.concat());
        assert!(!out.contains('\x1b'));
    }

    /// `tracing` may hand a line over in several writes.
    #[test]
    fn a_line_split_across_writes_is_still_recognised() {
        let out = run(&["12:00:00 INFO ⛏  8.3", " kH/s\n"], true);
        assert!(out.contains("\x1b[1;36m"));
        assert_eq!(out.matches("\x1b[1;36m").count(), 1, "tinted twice: {out:?}");
    }

    /// The reset must land before the newline, or the colour bleeds.
    #[test]
    fn the_colour_never_bleeds_into_the_next_line() {
        let out = run(&["12:00:00 INFO ⛏  8.3 kH/s\n"], true);
        assert!(out.ends_with("\x1b[0m\n"), "reset is misplaced: {out:?}");
    }
}
