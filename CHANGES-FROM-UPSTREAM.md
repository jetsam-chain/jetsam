# Changes from upstream

Jetsam is a derivative work of [Parano1d](https://github.com/ignotusnemo/parano1d)
v1.0.0, used under the Apache License 2.0. This file satisfies section 4(b) of
that license: *"You must cause any modified files to carry prominent notices
stating that You changed the files."*

Upstream baseline: commit `ffbb5ef` in this repository is the unmodified
upstream tree. Every change below is visible as a diff against it.

**Jetsam is an independent network.** It is not endorsed by, affiliated with, or
supported by Paranoid Zero or Ignotus Nemo. Jetsam coins have no relationship to
Parano1d (NOID) coins, and the two networks reject each other's peers.

---

## 1. Monetary policy — rewritten

| | Upstream (Parano1d) | Jetsam |
|---|---|---|
| Halving trigger | state expansion (`log_slots += 1` at 75% occupancy) | **block height** |
| Maximum supply | **none** — 1 NOID/block floor forever | **21 000 000 JTM, hard-capped** |
| Reward floor | 1 NOID/block, perpetual | **none** — emission reaches zero |
| Block interval | 20 s | **90 s** |
| Base reward | 50 NOID | 50 JTM |
| Development allocation | 3 years | **2 years** |

**Rationale.** Upstream's halving fires on state expansion, which requires
~12.6M occupied UTXO slots to trigger once. On a network without that many
users it never fires, so the effective emission is 50 NOID per block forever —
approximately 78.8M NOID per year, unbounded. Jetsam replaces this with a
height-based schedule under a hard cap.

The deterministic state-growth burn is **retained unchanged**: state growth
remains priced even though the halving no longer tracks it.

Emission schedule (90 s blocks, 28 800 blocks/month):

```
height           0 →      28 800   50      JTM/block
height      28 800 →     172 800   25      JTM/block
height     172 800 →     831 800   12.5    JTM/block
height     831 800 →   1 490 800   6.25    JTM/block
height   1 490 800 →   2 149 800   3.125   JTM/block
height   2 149 800 →   2 808 800   1.5625  JTM/block
height   2 808 800 →   3 467 664   0.78125 JTM/block
height   3 467 664 →         ...   0       (emission ends, ≈9.89 years)
```

The final tier ends at `EMISSION_END_HEIGHT` = 3 467 664, not at the naive
seventh boundary 3 467 800: the last 136 blocks are trimmed so that the sum of
`block_reward(h)` over every coinbase-carrying height (h ≥ 1 — genesis has no
coinbase) equals 21 000 000 JTM exactly. The cap is enforced by the height
schedule itself; no cumulative issuance counter exists.

## 2. Consensus — reorg fairness

Upstream measurements on the live Parano1d mainnet showed a miner producing
~35% of blocks retaining only ~3% of them: a minority miner is penalised far
beyond its hashrate share. Three compounding causes were identified, and Jetsam
addresses them:

- **Difficulty target anchored on the parent**, not on the block's own
  timestamp. Upstream derives the ASERT target from `header.timestamp`, so
  dating a block at `parent + 1` yields a heavier block at no computational
  cost — free weight. Jetsam removes this grinding vector.
- **Block interval raised to 90 s**, above the recursive prover's 7–35 s
  window. Upstream's 20 s interval leaves miners idle for a large fraction of
  every block (measured duty cycle: 79%) and hands a structural head start to
  whoever found the previous block.
- **Finality depth reduced** from 18 to 8 blocks, keeping wall-clock finality
  at ~12 minutes despite the longer interval.

## 3. Proof of work — TowerHash

The Poseidon2b permutation is **structurally unmodified**. Jetsam changes only
its instantiation and the sponge schedule around it:

- new round constants, generated with the upstream Poseidon2b reference
  generator from a documented nothing-up-my-sleeve seed;
- new domain separation tags;
- **nonce lane moved to index 0**, which removes the midstate precomputation
  available upstream (nonce at lane 10 of 16) and makes every attempt run the
  full 8 permutations.

These changes make Jetsam and Parano1d hashes mutually incompatible: neither
network's miner can produce valid work for the other.

## 4. Chain identity

- crates renamed `noid_*` → `jetsam_*`
- address HRP `o` → `j` (bech32m addresses become `j1…`)
- network magic `NOID` → `JTM1`, and every stream/file magic re-prefixed
  (leading byte `J`) so none is byte-identical to upstream's
- default ports 9600/9601 (upstream) → 9700/9701, new protocol id
  (`/jetsam/mainnet/1.0.0`), new genesis
- chain identity consolidated into `consensus/identity.rs`, which is the
  consumed source of truth: `NetworkConfig` takes its ports and protocol
  namespace from it, the address HRP is re-exposed from the codec constant in
  `jetsam_poseidon2b`, and the node's default `~/.jetsam` paths derive from its
  `DATA_DIR_NAME`

Domain separation tags are deliberately **not** derived from the coin name, so
that a rebrand can never alter consensus.

## 5. Upstream patches folded in

Local patches previously carried out-of-tree against Parano1d are integrated
here as defaults rather than environment flags:

- template overlap (proving the next template while still mining the current)
- the `NOID_TEMPLATE_NOGATE` behaviour, which upstream requires as an
  environment variable and without which the block template dies, is now the
  DEFAULT: templates are fabricated whenever initial sync is complete and one
  healthy authenticated peer exists, without closing during a competing
  block's P2P fetch. `JETSAM_TEMPLATE_STRICT_GATE=1` restores the historical
  strict gate

## Not changed

- the recursive `HistoryStep` architecture and its soundness argument
- the FRI-Binius proving stack
- the transparent (non-private) transaction model
- zero premine; the genesis block carries no coinbase at all — emission starts
  at height 1
