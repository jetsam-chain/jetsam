# libmdbx 0.6.6 — one-line fix for the `x86_64-pc-windows-gnu` target

This is the published `libmdbx` 0.6.6 crate with a single change, vendored here
so that the Windows binaries can be rebuilt by anyone from a clone of this
repository. Before it lived here, the Windows node could only be produced on
one machine, and nobody outside could check that the published `.exe` matched
the published source.

## The change

`src/flags.rs`, two `#[cfg]` attributes:

```rust
-#[cfg(windows)]              pub const fn c_enum(v: u32) -> i32 { v as i32 }
-#[cfg(not(windows))]         pub const fn c_enum(v: u32) -> u32 { v }
+#[cfg(all(windows, target_env = "msvc"))]        // -> i32
+#[cfg(not(all(windows, target_env = "msvc")))]   // -> u32
```

Nothing else differs from the crate as published — verified with
`diff -rq` against `~/.cargo/registry/src/*/libmdbx-0.6.6`.

## Why the original is wrong

`bindgen` maps a C enum to whatever underlying type the target's C compiler
uses: `int` (`i32`) under MSVC, `unsigned int` (`u32`) under mingw's GNU
toolchain. The crate keyed its shim on `windows` alone, which silently assumes
every Windows build is an MSVC build. Jetsam cross-compiles Windows from Linux
with mingw, so the published crate does not compile for that target at all.

Upstream issue: https://github.com/rust-lang/rust-bindgen/issues/1907

## Effect on the Linux build

None. On Linux, `not(all(windows, msvc))` selects the same arm as
`not(windows)`, so the generated code is identical. The patch changes the
`x86_64-pc-windows-gnu` target and nothing else.

## Refreshing it

Copy the published crate over this directory, reapply the two attributes above,
and re-run `diff -rq` to confirm `src/flags.rs` is the only file that differs.
