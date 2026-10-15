# Network economics

Jetsam fixes issuance by block height under a hard cap of 21,000,000 JTM and
prices persistent State growth separately from ordinary transaction work.

## Unit

```text
1 JTM = 1,000,000 μJTM
```

JTM is the currency ticker; the wallet uses ① as its interface symbol.
All consensus amounts are integers in μJTM.

## Block reward

The starting subsidy is 50 JTM. It halves at seven fixed heights and reaches
exactly zero; there is no floor (`jetsam_chain/src/consensus/emission.rs`):

| Height range | Block reward |
|---:|---:|
| 1 – 28,799 | 50.000000 JTM |
| 28,800 – 172,799 | 25.000000 JTM |
| 172,800 – 831,799 | 12.500000 JTM |
| 831,800 – 1,490,799 | 6.250000 JTM |
| 1,490,800 – 2,149,799 | 3.125000 JTM |
| 2,149,800 – 2,808,799 | 1.562500 JTM |
| 2,808,800 – 3,467,663 | 0.781250 JTM |
| 3,467,664 and above | 0 |

The last era is trimmed by 136 blocks so that the sum over every
coinbase-carrying height is exactly 21,000,000 JTM. Genesis carries no
coinbase. Halvings are counted in blocks: since block 38,400 a day holds 480
blocks instead of 960, so each halving arrives later in calendar time and the
daily issuance halved at that height. State expansion (`log_slots`) no longer
touches the subsidy; it still prices the State-growth burn.

## Launch development allocation

For the first two target-time years — blocks 1 through 369,600 — each block
subsidy is divided:

- 90% to the miner;
- 5% to the Network Fund;
- 5% to the Lab Fund.

There is no premine. From height 369,601, the complete block subsidy goes to
the miner. Summed over the schedule, the allocation is 747,000 JTM — 3.56% of
the maximum supply (`jetsam_chain/src/consensus/development_allocation.rs`).

To avoid creating two extra live UTXOs in every block, the two development
shares are paid in one mandatory two-output system record once per target-time
day: every 960 blocks up to and including block 38,400, then every 480 blocks.
The amount uses the reward tier active at that payout height.

The payout schedule, recipients and amounts are derived statelessly from height
and are proved inside `HistoryStep`. A miner cannot omit,
redirect or defer a due payout.

## Fees

The minimum transaction fee consists of:

- 5,000 μJTM per logical transaction;
- 100 μJTM per live input;
- 700 μJTM per live output;
- 2,500 μJTM times occupancy pressure for each net-new live slot.

Pressure multipliers are:

| Parent-State occupancy | Multiplier |
|---:|---:|
| Below 50% | 1× |
| 50% to below 75% | 2× |
| 75% to below 90% | 4× |
| 90% and above | 8× |

The State-growth component is burned. Base, input, output and voluntary tip
components are claimable by the miner.

A consolidation that turns several inputs into one output shrinks State and
pays no growth burn.

## Dynamic relay floor

The default mempool relays a transaction only when its fee satisfies both
consensus minimum and current relay policy. The dynamic floor is the greater
of 5,000 μJTM and 90% of the median fee among the last 50 transactions
admitted to that node's mempool.

This relay floor is local policy. Consensus fee accounting remains
deterministic from the transaction shape and parent-State occupancy.
