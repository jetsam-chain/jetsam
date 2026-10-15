# Protocol parameter reference

## Core

```text
block target            180 seconds since block 38,400 (90 before)
ASERT epoch               6 blocks
ASERT half-life         540 seconds
median-time-past         11 headers
future drift            120 seconds
transaction epoch        32 blocks
hard finality             8 blocks
```

## State

```text
initial domain           2^24 slots
maximum domain           2^32 slots
segment                  2^16 slots
expansion threshold      75%
finalized sample         18 headers
expansion majority       10 / 18
```

## Transactions and blocks

```text
Tx8x2 page               8 inputs / 2 outputs / 323 bytes
PagedSpend               1..128 pages
logical input cap        1,020
logical output cap       256
block body cap           256 fixed bodies
ordinary user pages      255
payout-block user pages  254
block user outputs       510
touched segments         256
```

## Proof system

```text
committed trace field    GF(2^128)
wide challenge field     GF(2^256)
challenge support        2^255
Poseidon2b width         4
S-box                    x^7
full rounds              8
partial rounds           58
B25 (small class)        m=23 since block 38,400 (m=22 before), up to 24 positions (25 before block 17,750)
B255                     m=25 since block 38,400 (m=24 before), up to 255 positions
wallet queries           65
History/BaseFold queries 133
B25 codeword             2^19 at rate 1/4
B255 codeword            2^21 at rate 1/4
serialized authorization max 92,696 bytes
target FRI security      128 bits
Block–Tiwari provable    127 bits
Block–Tiwari conjectured 127 bits
ideal-QROM boundary      64.707407428576 bits
NIST Post-Quantum Cryptography Category 1
Category 1 gate-depth floor 173.273866314232 bits
```

## Proof of work

```text
seed                     TowerHash, Poseidon2b sponge over 16 fields
digest from block 24,846 TowerWalk(seed)
digest before 24,846     seed
scratchpad               65,536 x 64-bit cells = 512 KiB
walk                     4 lanes x 131,072 rounds = 524,288 reads
Poseidon2b folds         33 per attempt
v1.4 anchor target       2^235
```

## Monetary

```text
1 JTM                   1,000,000 μJTM
initial subsidy          50 JTM
halvings                 by height, 7 (28,800 · 172,800 · 831,800 · 1,490,800 · 2,149,800 · 2,808,800)
emission end             height 3,467,664 (reward 0 from there; no floor)
maximum supply           21,000,000 JTM
development allocation   10% of each subsidy, blocks 1–369,600; 747,000 JTM in all
development payout       every 960 blocks up to 38,400, then every 480
base fee                 5,000 μJTM
input fee                100 μJTM
output fee               700 μJTM
base growth fee          2,500 μJTM / net-new slot
```

The explanatory and boundary rules are in
[Consensus parameters](../protocol/parameters.md).
