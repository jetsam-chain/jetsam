# Proof of work

Jetsam proof of work runs after the nonce-independent `HistoryStep` has been
proved. It has two stages:

1. **TowerHash** — a domain-separated Poseidon2b sponge over the fixed semantic
   header fields and the nonce — produces a 32-byte seed;
2. **TowerWalk** — a cache-resident walk over a 512 KiB scratchpad — turns that
   seed into the digest compared with the target.

The walk is part of consensus from block 24,846 on the public network
(activated on 29 September 2026 at 18:21:52 UTC). Below that height the digest
is the TowerHash output alone, so every earlier block keeps its original proof
of work. TowerWalk is a CPU proof of work that lives in each core's L2 cache;
the GPU miners used before block 24,846 no longer work.

## Field schedule

The `POWHDR__` sponge absorbs exactly 16 `GF(2^128)` elements:

| Index | Field |
|---:|---|
| 0 | 128-bit nonce |
| 1–2 | `prev_block_hash` |
| 3–4 | `state_root` |
| 5–6 | `tx_root` |
| 7 | `timestamp` |
| 8 | `height` |
| 9–10 | `miner_address` |
| 11–12 | `difficulty_target` |
| 13 | `log_slots` |
| 14 | `active_slot_count` |
| 15 | `alloc_counter` |

The nonce is field 0 (`POW_NONCE_FIELD_INDEX`), so it enters the first
permutation and no sponge state can be precomputed across nonces.

Thirty-two-byte values are split into two little-endian 128-bit halves. Scalar
integers are zero-extended. With sponge rate two, the schedule is exactly eight
rate blocks and needs no variable-length padding.

The PoW domain differs from both nonce-bearing block identity and the
nonce-free semantic header commitment.

## TowerWalk

From block 24,846 the TowerHash output is a seed, not the digest. TowerWalk
reads it as four little-endian 64-bit words and:

1. fills a scratchpad of 65,536 cells of 64 bits (512 KiB) with a
   multiply-xorshift sequence derived from the seed, folding the lane state
   through Poseidon2b every 4,096 cells;
2. walks the scratchpad for 131,072 rounds of four lanes — 524,288
   data-dependent reads — where each lane reads the cell its own state
   addresses, mixes the value in and writes the result back to that cell;
   lanes run in order within a round, so a lane reads what the previous lane
   has just written;
3. folds the lane state through Poseidon2b every 8,192 rounds and once at the
   end, 33 folds in all, and returns the four lane words as the 32-byte digest.

All arithmetic wraps modulo 2^64. The constants are fixed by consensus:

| Constant | Value |
|---|---:|
| Scratchpad cells (`CELLS`) | 65,536 (512 KiB) |
| Lanes (`LANES`) | 4 |
| Walk rounds (`ROUNDS`) | 131,072 |
| Fold period in the walk (`PERM_PERIOD`) | 8,192 rounds |
| Fold period in the fill (`FILL_PERM_PERIOD`) | 4,096 cells |
| Mix multiplier (`MULT_C`) | `0x9E3779B97F4A7C15` |
| Mix xorshift (`XORSHIFT`) | 29 |

The scratchpad is sized to one core's private L2 cache. The function lives in
`jetsam_poseidon2b::towerwalk`, shared by the node and the external miner so
that there is one implementation of the consensus hash. A bit-level
specification, a dependency-free reference implementation and 256 frozen test
vectors are in the
[mining specification](../mining/stratum.md#310-towerwalk-the-digest-from-block-24846).

## Target comparison

The digest and target are interpreted as 256-bit little-endian integers. A
nonce is valid only when:

```text
seed       = TowerHash(fields)
pow_digest = TowerWalk(seed)     at height ≥ 24,846
pow_digest = seed                below height 24,846
pow_digest < difficulty_target
```

Equality fails.

## ASERT

The target interval between accepted blocks is 90 seconds. Proof preparation,
nonce search and propagation all occupy that interval. ASERT uses a six-block
reference epoch and a 120-second half-life. At each height, validation derives
the exact target from the canonical anchor, elapsed time and height delta.

A timestamp must also be greater than median time past over the previous 11
headers and no more than 120 seconds ahead of the validating node's wall clock.

The first TowerWalk block could not inherit the sponge-era target: a walked
attempt costs far more than a sponge attempt, and ASERT anchors on the
parent's timestamp, so it absorbs a loss of hashrate but not a stall. Block
24,846 therefore carries a fixed anchor target of `2^235`, set on the easy
side; ASERT tightens from there towards the 90-second interval.

## Runtime kernels

The same fixed permutation is evaluated in packed nonce batches. The production
binary selects the best supported implementation at runtime:

- `pclmul` baseline on x86-64;
- `avx2+vpclmul` where available;
- `avx512bw+vpclmul` on supported hosts;
- `neon+pmull` on ARM64.

Packed execution changes throughput, not the digest. The scalar implementation
is a test oracle and is not a production fallback.

These kernels accelerate the Poseidon2b permutation. The walk itself is scalar 64-bit arithmetic whose speed is set by the
latency of reads into the 512 KiB scratchpad. Each mining thread holds its own
scratchpad, so one thread per physical core is usually the best setting; size
`--cpu-threads` on the `walked digest` rate that `jetsam --bench` prints.

## External mining boundary

An external worker receives the exact 16-field schedule, nonce index, target
and a `pow_walk` flag. When `pow_walk` is `true` the worker must search
`TowerWalk(TowerHash(fields))`; absent or `false` means the TowerHash digest
alone. The node sets the flag from the template's own height; a worker must
read it and never derive it from the height. It returns only a canonical
16-byte little-endian nonce. The node checks it against the immutable,
single-use template, with the digest that height requires, before committing a
block.

`jetsam-miner` declares that it can walk with the HTTP header
`X-Jetsam-PoW: walk`. A node ignores the header; a pool serving TowerWalk work
uses it to tell walking miners from pre-fork ones, which cannot find a block.

The worker cannot alter a transaction, State root, payout or proof. A solved
nonce for an expired or stale template is rejected.
