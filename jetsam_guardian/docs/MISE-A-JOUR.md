# Updating the guardian software (T5.5)

Reference: `jetsam-v17-gardien-logiciel_DESIGN.md` §7 (decision D37).

## Build

- Toolchain pinned by `rust-toolchain.toml` (1.96.0); dependencies from the local registry,
  `cargo build --locked --offline --release -p jetsam_guardian`.
- `--remap-path-prefix` of the source and registry paths, `SOURCE_DATE_EPOCH` set, `CARGO_INCREMENTAL=0`:
  `scripts/reproducible-build.sh <source dir> <target dir>` (in this crate) does it.
- Two builds on two hosts; the SHA-256 of the four binaries must be identical, otherwise stop. One of the
  two hosts is ideally not administered from the agents' workstation.
- Publication: version, commit, the SHA-256 of `jg-observe`, `jg-net`, `jg-sign`, `jg-aggregate` and of
  the node, a summary of the diff, the audit reference.

## Install on a guardian machine

- Binaries in a read-only directory owned by root, one directory per version.
- `allowed.sha256` (state directory) is edited BY HAND, on the machine, by the operator: `jg-start`
  refuses any binary or configuration whose hash is absent.
- No automatic update, no code downloaded at run time. Going back = stop by PID, start the previous binary
  (still in the allow-list).

## Staged deployment (D37)

| Stage | Machines on the new version | Minimum | To go on |
|---|---|---|---|
| 1 | 2 | 48 h | 0 HALT; partials identical to the old version's for the same statements; same refusal codes; no outgoing connection outside the configured destinations |
| 2 | **4 at most** | 7 days, and until the audit of the diff (any change of `jg-sign` or of the statement format) and a GO | same |
| 3 | 7 | — | — |

Why 4 at most: a version that would sign lies holds only 4 partials; the 3 others sign only what their own
node shows. An urgent security fix passes on a GO but stays on ≤ 4 machines for ≥ 24 h. `jg-sign` is
updated separately from `jg-observe` / `jg-net`, and less often. The node of the guardian machines follows
the same stages (never more than 4 changed at once).

## Measured in the rehearsal

See the P5 report: two release builds of the same commit from two source directories on one host (the
second host is a machine of the user: not done without G10).
