# Local patch: bounded PRUNE backoff

Vendored copy of `libp2p-gossipsub` 0.48.0 (our resolved version, via
`libp2p` 0.55.0) with one change: `BackoffStorage::update_backoff`
(`src/backoff.rs`) clamps the backoff duration to `MAX_BACKOFF` (1 hour)
before using it, instead of accepting an attacker-controlled value
unbounded.

## Why

Two upstream advisories, fixed in 0.49.3 and 0.49.4 respectively:

- GHSA-gc42-3jg7-rxr2 / CVE-2026-33040 — a PRUNE control message carrying
  a very large `backoff` field (sent by any connected peer, no
  subscription or mesh membership required) makes `update_backoff` compute
  `Instant::now() + time` with an unbounded `time`, overflowing `Instant`
  and panicking the task handling that peer.
- GHSA-xqmp-fxgv-xvq5 / CVE-2026-34219 — a backoff value just under the
  overflow threshold passes insertion but later overflows in
  `BackoffStorage::heartbeat()`'s `backoff_time + slack > now` expiry
  check, on a subsequent heartbeat tick.

Both overflows are reached only through values stored by `update_backoff`.
Clamping there closes both: every `Instant` kept in `BackoffStorage` is at
most `now + 1h`, and the heartbeat's `+ slack` addition (at most a few
seconds at our `heartbeat_interval`/`backoff_slack`) stays far below
`Instant`'s representable range.

## Why 1 hour

Our own config (`jetsam_p2p/src/behaviour.rs`) does not override
`prune_backoff` (default 60s) or `unsubscribe_backoff` (default 10s), and
a legitimate peer has no reason to request an hour-long backoff. 3600s is
generous headroom above anything we or a well-behaved peer would ever set,
while staying nowhere near overflow territory.

## Upgrade path

This is a stopgap, not a replacement for upgrading. When `libp2p` ships a
version pulling in `libp2p-gossipsub` >= 0.49.4, drop this patch and the
`[patch.crates-io]` entry in the root `Cargo.toml`, bump the `libp2p`
version requirement, and re-run the full workspace test suite before
removing this directory.

## Scope

Only `src/backoff.rs` differs from the upstream 0.48.0 source on
crates.io. No other file, no protocol behavior, no wire format, and no
consensus-relevant code is touched.
