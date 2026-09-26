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
                // Re-arm after every reset the formatter itself emitted.
                //
                // `tracing` styles the timestamp and the level and turns
                // styling off after each. Colouring once at the head of the
                // line therefore ends a dozen bytes in: the clock came out
                // tinted and the rate — the only part worth reading — arrived
                // plain. Shipped that way in v1.4.2, because every test in
                // this module fed it a line with no escape codes in it.
                let mut rest = body;
                while let Some(at) = find(rest, OFF) {
                    let after = at + OFF.len();
                    self.inner.write_all(&rest[..after])?;
                    rest = &rest[after..];
                    // Nothing left to colour: do not leave a dangling code.
                    if !rest.is_empty() {
                        self.inner.write_all(on)?;
                    }
                }
                self.inner.write_all(rest)?;
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
    find(haystack, needle).is_some()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
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

    /// A real line, as `tracing` hands it over — it styles the clock and the
    /// level and turns styling **off** after each.
    const REAL: &str = "\x1b[2m00:25:12\x1b[0m \x1b[32m INFO\x1b[0m ⛏  2.96 kH/s  ·  \
                        4 threads · 1000636 hashes total \x1b[3mheight\x1b[0m\x1b[2m=\x1b[0m6906\n";

    /// Is `needle` inside an active colour span, or did a reset end it first?
    fn is_tinted(out: &str, needle: &str) -> bool {
        let at = out.find(needle).expect("needle absent");
        let mut armed = false;
        let mut rest = &out[..at];
        while let Some(esc) = rest.find('\x1b') {
            let tail = &rest[esc..];
            let end = tail.find('m').map(|i| i + 1).unwrap_or(tail.len());
            armed = &tail[..end] != "\x1b[0m";
            rest = &tail[end..];
        }
        armed
    }

    /// What every earlier test in this module missed, and what shipped broken:
    /// colouring from the head of the line dies at the formatter's first
    /// reset. The clock came out tinted and the rate — the only part worth
    /// reading — arrived plain.
    #[test]
    fn a_real_line_stays_coloured_past_the_formatters_own_resets() {
        let out = run(&[REAL], true);
        assert!(
            is_tinted(&out, "2.96 kH/s"),
            "the rate is not inside a colour span: {out:?}"
        );
        assert!(
            is_tinted(&out, "hashes total"),
            "the tail of the message lost the colour: {out:?}"
        );
        assert!(
            is_tinted(&out, "6906"),
            "the field values lost the colour: {out:?}"
        );
    }

    /// Re-arming must not survive the line it belongs to.
    #[test]
    fn a_real_line_still_ends_reset() {
        let out = run(&[REAL], true);
        assert!(out.ends_with("\x1b[0m\n"), "reset is misplaced: {out:?}");
        assert!(!out.contains("\x1b[0m\x1b[1;36m\n"), "re-armed for nothing: {out:?}");
    }

    /// An ordinary line keeps the formatter's own colours untouched.
    #[test]
    fn an_ordinary_line_is_passed_through_byte_for_byte() {
        let plain = "\x1b[2m00:25:12\x1b[0m \x1b[32m INFO\x1b[0m peer connected\n";
        assert_eq!(run(&[plain], true), plain);
    }
}
