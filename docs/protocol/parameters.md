# Consensus parameters

This page groups the launch consensus constants. Values are integers unless
stated otherwise.

## Time and finality

| Parameter | Value |
|---|---:|
| Mainnet genesis timestamp | 2026-08-21 16:00:00 UTC (`1787328000`) |
| Target block interval | 180 seconds since block 38,400 (90 seconds before) |
| ASERT reference epoch | 6 blocks |
| ASERT half-life | 540 seconds |
| Median-time-past window | 11 headers |
| Maximum future timestamp drift | 120 seconds |
| Transaction epoch | 32 blocks |
| Hard-finality depth | 8 blocks |
| Maximum accepted reorganization | 8 blocks |
| Recent block-body retention | 8 blocks |
| Undo retention | 16 blocks |

## Block limits

| Parameter | Value |
|---|---:|
| Fixed bodies including system records | 256 |
| User page positions | 255 |
| User page positions on payout block | 254 |
| Live user inputs | 1,020 |
| Live user outputs | 510 |
| Live user actions | 1,530 |
| Distinct State segments touched | 256 |
| Transaction-tree leaves | 256 |

## Transaction limits

| Parameter | Value |
|---|---:|
| Physical page encoding | 323 bytes |
| Inputs per page | 8 |
| Outputs per page | 2 |
| Pages per logical spend | 1–128 |
| Inputs per logical spend | 1–1,020 |
| Outputs per logical spend | 0–256 |
| Authorization wire cap | 256 KiB |
| Logical intent wire cap | 303,495 bytes |

## State

| Parameter | Value |
|---|---:|
| Initial `log_slots` | 24 |
| Maximum `log_slots` | 32 |
| Slots per segment | `2^16` |
| Expansion occupancy | 75% |
| Expansion finalized window | 18 headers |
| Required high-occupancy headers | 10 of 18 |

## Proof classes

| Class | Dimension before block 38,400 | Dimension from block 38,400 | Effective page positions before block 17,750 | From block 17,750 |
|---|---:|---:|---:|---:|
| Small (`B25`) | 22 | 23 | 0–25 | 0–24 |
| B255 | 24 | 25 | 26–255 | 25–255 |

The primary reward is excluded from effective page-position count. A live
development payout counts as one position.

## Proof of work

| Parameter | Value |
|---|---:|
| Header schedule | 16 `GF(2^128)` fields, nonce in field 0 |
| Seed | TowerHash (Poseidon2b sponge, 8 permutations) |
| TowerWalk activation (public network) | block 24,846 |
| TowerWalk scratchpad | 65,536 × 64-bit cells (512 KiB) |
| TowerWalk lanes × rounds | 4 × 131,072 (524,288 dependent reads) |
| Poseidon2b folds per attempt | 33 |
| Anchor target of the first walked block | `2^235` |

## Monetary

| Parameter | Value |
|---|---:|
| Atomic unit | 1 μJTM |
| Units per JTM | 1,000,000 |
| Starting subsidy | 50 JTM |
| Subsidy floor | none — the reward is 0 from height 3,467,664 |
| Halvings | 7, by height, the last one trimmed so the sum is exactly 21,000,000 JTM |
| Base transaction fee | 5,000 μJTM |
| Fee per live input | 100 μJTM |
| Fee per live output | 700 μJTM |
| Base growth fee per net-new slot | 2,500 μJTM |
| Development allocation period | blocks 1–369,600 (two target-time years; 747,000 JTM in all) |
| Development payout interval | 960 blocks up to and including block 38,400, then 480 blocks |

## Storage and network

| Parameter | Value |
|---|---:|
| Header encoding | 212 bytes |
| P2P TCP port | 9700 |
| Local RPC TCP port | 9701 |
| Direct-sync header request | 512 headers |
| Header protocol batch cap | 4,096 headers |
| Mempool transaction count | 1,024 |
| Mempool intent-byte budget | 384 MiB |
| Peer-store entries | 500 |
| External template lifetime | 120 seconds |
