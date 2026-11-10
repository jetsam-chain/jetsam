# Guardian keys: generation, backup, rotation, revocation (T5.4)

Private document of the project. Reference: `jetsam-v17-gardien-logiciel_DESIGN.md` §6 (decisions D34, D35, D38, D39).

## What exists

| Key | Where | Protects | If stolen |
|---|---|---|---|
| Group key (BLS12-381 G1, 48 bytes) | public: `params.rs`, the contract's constructor | nothing (public) | — |
| 7 shares (one scalar each, Shamir 5 of 7) | each guardian machine, `sign/share-g<gen>.sealed`, sealed for that machine's key | the fast path's co-signatures | ≤ 4 shares: nothing; ≥ 5 of one generation: the project's signature (bounded by the contract caps) |
| Machine key (X25519) | each guardian machine, `sign/machine-g<gen>.key.sealed`, sealed by the operator's passphrase (Argon2id 256 MiB, 3 passes) | the share at rest | with the passphrase: that machine's share |
| Feed key (Ed25519) | `net/feed.key` | authenticity of the heartbeat and observation lines | forged `obs` lines for one guardian (a source of SG-10 among others) |
| Freeze key (ECDSA, L1) | hardware wallet held by the user (D39), never on a guardian machine nor on the agents' workstation | freezing the mint paths ≤ 48 h, ≤ 96 h / 30 days | delays only |
| Revocation certificate (0x21 body, height 2^63) | ONE sealed copy, offline, at a place held by the user, distinct from the two backup places (D34, A1 m7) | killing the fast path for ever if ≥ 5 shares are compromised | the fast path dies (no theft); before the bridge genesis it does nothing on the L1 |

## Ceremony (offline machine, RAM only)

1. Boot a read-only medium whose SHA-256 is checked; no network hardware; no swap.
2. Check the SHA-256 of `jg-ceremony` (reproducible build, `MISE-A-JOUR.md`).
3. On each guardian machine: `jg-keys machine-init --state <dir> --generation <g> --passphrase-fd 3`
   (passphrase typed ON THAT MACHINE, never through the agents' workstation, D38). Compare the printed
   fingerprint on paper.
4. `jg-ceremony run --machine-keys <7 public keys> --out <usb> --dice-fd 3 --network <id> --l1-chain-id <n> --l1-bridge <addr>`
   with at least 100 dice throws on descriptor 3. It checks every share against the commitments, the 21
   subsets of 5 against the direct signature of the test message (outside the statement domain), and that
   4 partials fail. It writes only sealed shares, `public.json` and the certificate.
5. Print: group key (48 and 128 bytes), commitments, share keys, sealed files' SHA-256, test message and
   signature, the certificate's SHA-256. Publish the group key and the commitments.
6. Power off. The group secret existed in RAM only.
7. On each machine: `jg-keys import --config guardian.toml --sealed <its file> --passphrase-fd 3 --h0-header <hex>`:
   opens the share, checks it against the commitments, signs the test message and checks it against
   `pk_i`, writes the journal's GENESIS with the trust anchor H0 (confirmed by two other guardians).

Any sign of compromise during the ceremony: start again and destroy the shares.

## Backups (D35, default 3 + 2)

Five sealed shares on two offline media: three at place A, two at place B; each medium encrypted with its
own passphrase, the passphrase of A kept at B and conversely (paper). Never on the agents' workstation,
never on a guardian machine, NEVER in `/root/wallet/` (that rule is for wallets; applied to a share it
would recreate the common path K6). Variant 2 + 2: theft of both places is not enough, loss of the 7
machines loses the key.

## Rotation: re-sharing at a constant group key

When: a machine replaced, ≤ 4 shares suspected, and every 6 months. The group key never changes (it is
pinned in `params.rs` and in the contract).
1. New machine keys `machine-init --generation g+1` on the 7 machines.
2. Stop `jg-sign` by PID on ≥ 5 machines; export their shares to the ceremony machine's ephemeral key.
3. Offline: rebuild the secret (Lagrange), check it equals `C_0`, deal a new polynomial with the same
   constant term (`ceremony::reshare`), new commitments with the same `C_0`.
4. Import with a `RESHARE` record; the journal continues (the history belongs to the group key).
5. Destroy the old shares, machine keys and backup media: 5 old shares still rebuild the same secret.
Aggregators then reject the partials of the old generation.

## Revocation and incidents (design §6.6)

| Situation | Action |
|---|---|
| 1-2 shares suspected | stop that signer by PID; re-share within 72 h |
| 3-4 shares suspected | freeze (≤ 48 h) if an attempt is plausible; re-share within 24 h |
| ≥ 5 shares, a common admin path compromised, or a foreign finalization | on the user's GO (irreversible): publish the certificate on the L1 (`proveEquivocation`) and on Jetsam (object 0x21) |

Voluntary revocation: while it still holds 5 shares, the project may sign two contradictory statements
itself and submit the proof.

## Gas token of the L1 accounts (A1 m8)

The freeze key, the relayer and the aggregators hold the L1's gas token. Their funding goes through a NEW
path, never through an account or platform tied to an identity or to another project of the user; each
funding is written here: origin (no identity data), date, amount.

| Date (UTC) | Account | Amount | Origin |
|---|---|---|---|
| — | — | — | — |

## Never

- type a passphrase through the agents' workstation; give an agent session root on a guardian machine;
- keep a share, a backup or the certificate on a machine that runs miners, rentals or a thermal shutdown;
- start anything at boot (no systemd unit, no cron, no rc.local);
- copy a share to `/root/wallet/`.

## Rehearsal (TEST keys)

The private rehearsal uses the T0.5 TEST dealing `fixture` (`jg-ceremony test --label fixture`), the group
key pinned in the node's `rehearsal-v1-7` profile, TEST passphrases in clear and `--test-kdf`. None of it
is ever used outside the private rehearsal.

## Bridge genesis key (D-AV1, D73; audit A5-8)

The contract's `genesisKey` installs the bridge genesis once (`initGenesis(cp)`) and has no power after it
(`test_genesis_key_has_no_power_after_genesis`). Before it, it is the bridge's single root of trust: a false
genesis (a block off the chain, or a private branch) kills the bridge or, with >= 40 % of the reference
hashrate, lets fictitious deposits be minted. Holder: **[NV]** (not recorded yet; to be written here).

Procedure (written down, nothing deployed by it):

1. The key is generated on the offline machine of the ceremony (RAM only), used once, never on a guardian
   machine, a mining machine or the agents' workstation.
2. The genesis checkpoint is printed by `jg-relay --config <relay.toml> print-genesis <L>` on **two** relayer
   hosts reading **different** nodes; the two `checkpoint_id` and `calldata` must be equal byte for byte, and
   `L` must be in `[J ; BRIDGE_DEPOSITS_FROM_HEIGHT]`.
3. The same `checkpoint_id` is pinned in the nodes' profile (`L1BridgeParams`, A5-8: no release is payable
   while the contract's `genesisId` (slot 13) is not the pinned one) BEFORE the profile's `L1_BRIDGE` is set
   (T8.3).
4. `initGenesis` is sent; the `BridgeGenesis(id, lockedHeight, lockedBlockId, lockedCumWork)` event and
   `genesisId()` are read back from two L1 endpoints and compared with step 2.
5. The key is then destroyed: the key file overwritten and deleted (`shred -u`), the RAM machine powered off; a
   multisig genesis key has its shares discarded the same way. Written here: date (UTC), who did it, the
   `genesisId`, the L1 transaction, the two hosts of step 2.

| Date (UTC) | genesisId | L1 tx | Hosts of step 2 | Destroyed by |
|---|---|---|---|---|
| — | — | — | — | — |

## Hot keys of `jg-fill` and `jg-relay` (decision 10 of the audit A5, A5-15)

Decision (user, 04/11, recommendation of the audit): the hot keys stay in a **clear file, mode 0600** (or
0400), owned by the uid of the process, so that the robot and the relayer restart without an operator; the
risk is bounded by the **stock** of the robot and by the gas of the relayer's account.

- **Key** (each): secp256k1, `[signer] key_file`, 64 hex characters in clear; refused unless 0600 or 0400 and
  owned by the process's uid (`HotKey::load`); never logged, never passed over IPC, its `Debug` shows the
  address only, zeroized when dropped. `jg-fill`: `key_file` must match `hot_address` of `fill.toml`.
- **Fees** (A5-13, `jg-fill` and `jg-relay` alike): off the private rehearsal `[signer] priority_fee_wei` is
  required (the L1 endpoint's `eth_maxPriorityFeePerGas` is never taken), at most `max_priority_fee_wei` (0.1 gwei
  by default since E-9, adapted to Base; replacements included), else no start. E-9: `jg-fill`'s transactions stay
  within `[signer] daily_gas_budget` (30 M gas per 24 h of the L1 clock by default, kept in `<state_dir>/gas.json`):
  above it a send is refused before signing, `ALERT GAS_BUDGET` journaled; `jg-relay` has none unless configured. `jg-fill` halts (`HALT_REPEATED_REVERTS`, latched) after 3 of
  its transactions reverted in a row, and keeps no refusal that depends on the L1 clock. `jg-relay` too (A5FIX4):
  3 of its own transactions reverted in a row, none explained by another one landing first (the tip moved, the
  deposit minted, another point finalized), latch `HALT_REPEATED_REVERTS` in `<state_dir>/HALT`; it then reads,
  learns, rebuilds, publishes `final.json` and queues, and sends nothing until an operator removes the file.
- **An advance in flight at a stop** (M8 of the re-audit n°2, A5FIX4): `jg-fill`'s signer never sends again an
  advance it finds unmined at a recovery (start, or a send after a receipt timeout): its rule `ALREADY_UNDER_A_POINT`
  was decided before the stop. Its nonce goes to a zero-value transfer of the hot key to itself (fees raised by 1/8,
  so that it replaces the advance in any pool still holding it); the orders are decided again at the next step.
- **Process** (A5-15): `jg-fill` and `jg-relay` harden themselves before reading anything
  (`jetsam_guardian::harden`): `PR_SET_DUMPABLE=0` and `RLIMIT_CORE=0` (they do not start otherwise);
  `mlockall` when `RLIMIT_MEMLOCK` is unlimited or the process has `CAP_IPC_LOCK`, else a warning — then the
  host's swap must be off or encrypted. Started by `ops/jg-start fill|relay` only (SHA-256 of the binary and of
  the configuration in `allowed.sha256`, one instance, PID file); `jg-fill` also holds a `flock` on its
  `state_dir` and refuses a `fill.toml` that is not 0600/0400 and owned by its uid (it names `refund_to`).
- **Stock** (robot): wJTM balance of the hot key ≤ `max_stock` (10 000 JTM by default since D123), checked at start (no
  start above it) and at every step (N9: above it, `PAUSE_STOCK_ABOVE_CAP` and an ALERT, nothing advanced until
  the excess is moved to `refund_to`; anyone can push the hot key over its cap by sending it wJTM — a gift that
  pauses the robot until the operator moves it); its own advances in flight stay within it (`outstanding_cap ≤
  max_stock`, D123), under the contract's 50 000 JTM outstanding and 50 000 JTM / 24 h per advancer. **The maximum
  loss per incident is the hot key's stock.** Repayments go to `refund_to`, a cold address
  other than the hot key.
- **Fee of an advance** (A5-1, contract revised: `ADVANCE_MIN_AGE` = `D_SLOW` = 24 h): an advance less than 24 h
  old at the mint of its deposit is repaid what it paid only, its fee going to the recipient. An advance made before
  the acceptance of the point that covers its deposit is always 24 h old at the mint; so `jg-fill` never advances a
  deposit at or below the locked height of the contract's tip (`ALREADY_UNDER_A_POINT`: the slow path pays the
  recipient within 24 h), and does not start unless the bridge's `ADVANCE_MIN_AGE()` is 86 400. **An advance
  doubled by the relayer** (its point covering the deposit lands in the few minutes between the robot's read of the
  tip and its advance) **loses its fee: accepted** (the recipient gets it; the robot is repaid what it paid).
- **The rent of an advance after its point (re-audit n°2, a known limit, written as is)**: an advance made after the
  acceptance of the point that covers its deposit is young at the earliest mint to the second only: minted one L1
  block (12 s) later, it keeps its 1 %. **A third party can still take 1 % on the deposits the robot leaves
  (`ALREADY_UNDER_A_POINT`), against 24 h of capital**, with no reorganization risk: about the service the robot
  renders, so fair competition, not a flaw (contract test
  `test_known_limit_an_advance_one_l1_block_after_the_point_keeps_its_fee_for_24h_of_capital`).
- Holder of each hot key and rotation procedure: **[NV]** (to be written here).

## Loss per incident (decision 11 of the audit A5)

What is said publicly: **the maximum loss per incident is the robot's stock (`max_stock`)**. The contract's
per-advancer outstanding cap (`ADVANCE_OUTSTANDING_CAP`, 50 000 JTM since D123) is a circuit breaker against a faulty
robot, not a property of the system; no figure is derived from a pool's dated share. The bridge assumes an
honest majority of the mining (a majority can prove deposits that do not exist, A5-21).

## Relayers (decision 12 of the audit A5, A5-9)

A second relayer is allowed only with a contract that carries the A5-4 fix (`MustExtendTip`: a point on the
tip's own chain that does not extend the tip is refused, the finality clock does not move): before it, two
honest relayers out of step restarted the 24 h clock. Two relayers then share nothing but the L1: a proof
made while the other's point landed is void (refused, made again on the new tip).

## Deposits in no root (re-audit N2, A5-22; C1 of the re-audit n°2)

Anyone may submit a point that leaves whole deposit blocks out (the program allows it; it never lets one
deposit of a block out, nor invents one). Since the second re-audit (C1, option B) the proof commits the heights
of the blocks it gave (public value 23) and the contract publishes them in `CheckpointAccepted`: `jg-relay`
rebuilds the tree of every point it learns from that list, without any search (which of its deposit blocks the
point left out: `ALERT POINT_LEAVES_DEPOSITS_OUT`; `ALERT POINT_TREE_UNKNOWN` only while a listed block's body is
not read here, or for a point of another chain), extends such a point when it is the contract's tip, finalizes it
all the same with `ALERT FINALIZING_POINT_NOT_COMPLETE` (those deposits are in no root of its branch for ever;
refusing would hold every other deposit), and names what is left with `ALERT DEPOSITS_IN_NO_FINAL_ROOT`: the
deposits its tree left out or, when the final point's tree is unknown here, **every deposit under it not minted
yet** (re-audit n°2, M3: never an empty list). A deposit in no root is minted only by `claimDepositDirect`:

1. **The relayer queues it by itself** (`<state_dir>/claim-direct/<height>-<txid>.req`, once: a request done is not
   queued again), the same queue as an operator's `jg-relay --config <relay.toml> claim-direct <height> <txid>`
   (which checks that the archive holds the body of block `<height>` with that deposit; it never touches the L1).
2. The running `jg-relay` (the only process signing with its key) proves the oldest request with `[prover]
   claim_command` (the deposit-claim program, input `JCLMIN03`: the whole archived block and the deposit's group,
   `{input}`, `{proof}`, `{public_values}` as for `command`), runs the program natively first, and sends
   `claimDepositDirect` against the final point's anchor root once block `<height>` is under it, within
   **`claim_direct_daily_gas`** (default 10 000 000 gas in 24 h of the L1 clock, about 22 direct claims; 450 000
   counted before each; 0: none; over budget: `ALERT CLAIM_DIRECT_BUDGET`, the request waits). Without
   `claim_command` the request waits, with an alert. Done or refused: `<height>-<txid>.done` (its outcome).
   A request whose block is not under the final point of this chain (above its L, at or below the bridge genesis,
   or an orphan's archived body: an operator's request may wait for ever) waits apart and never holds the requests
   behind it (R3-3 of the re-audit n°3).

**A final point whose tree is unknown here** (a block its list names not read yet, R3-6 of the re-audit n°3): the
deposits of the blocks it left out (named by no list on its branch) are queued at once; those of the blocks it gave
are in its root, one body away from a `claimDeposit` (≈ 81 k gas instead of ≈ 407 k): they wait for the tree up to
48 h after the point's acceptance (`TREE_WAIT_SECS`, 24 h after the earliest finality), then are queued for
`claimDepositDirect` as well (M3). **`acknowledge` (D41) of a block a third party's point lists** (R3-7) means this
relayer never rebuilds that point's tree, so never extends it: import the body instead whenever one exists.

**A point after a long stop** (R3-5 of the re-audit n°3): `submitCheckpoint` costs ≈ 750 gas per height listed;
`jg-relay` lowers the point's tip until its window lists at most `(max_gas / (1 + margin) − 400 000) / 800` deposit
blocks (2 500 with the default signer), the next point taking the rest. When that point does not advance or MN-5
refuses it (a stop long enough for the bound to bite: ≈ 2 500 blocks with a deposit each, more than 5 days):
`ALERT NO_SUBMISSION TooManyDepositBlocks(n)`, raise `[signer] max_gas`.

An advance of such a deposit is repaid by that mint as by any other (same key: txid, recipient, amount). A third
party who forces every deposit this way (by being first to extend at every cycle) costs the relayer one claim
proof and ≈ 0.4 M gas per deposit, bounded by the budget; the deposits wait beyond it, none is lost.

## Hashrate (audit A5-5, written publicly)

The bridge stops by itself when the mining collapses durably below about 15 % of its reference rate, and starts
again only when the hashrate returns: a point is accepted only if it lands within 1 800 s of its parent's tip
block or carries at least 40 % of the work its fork base's target produces in the time since (MN-5, kept on
purpose: a rate relative to the current target would be an attacker's lever). Burns and releases go on.

What a miner can do to the bridge, with MN-5 (re-audit n°2, §1.4; these replace the "5-10 %" written after A5FIX2):
**with about 40 % of the reference hashrate, freeze the finality (never steal)**: a real switch abandons the tip's
locked block, 10 blocks from its fork base within 4 500 s; **with about 30 %, decanonize points without freezing**:
shift the tip without abandoning its locked block, 2 blocks + a proof + its inclusion within 1 800 s of the fork
base (about 15 % with a public orphan exactly at the tip's height). Contract tests
`test_known_limit_shifting_the_tip_without_a_switch_must_land_within_1800s_of_its_base`,
`test_known_limit_a_real_switch_needs_ten_blocks_within_4500s_of_its_base`.

## Proofs and the address (A5-28)

Two programs fix the bridge's CREATE2 address, through the vkeys in `verifier` and `claimVerifier`: the checkpoint
program (`JSPVIN08`, 768 bytes) and the deposit-claim program (`JCLMIN03`, 192 bytes, now on the nominal path of the
deposits a point leaves out). **Neither has ever produced a real proof** (SP1 executor and TEST verifier only). **No
address is frozen before a real proof of each has been verified by the deployed verifier.**

## Robot and relayer on one host (re-audit, A5-7)

`jg-fill` advances only a deposit whose block body the relayer published (`<publish_dir>/blocks/`): the robot
reads that directory locally. Deploy the robot on the relayer's host, or give it the directory on a shared
file system it can only read.

## Rehearsal binaries (re-audit N5)

The format of `BridgeAccountProof` (A5-8: the account and the storage of slot 13) is not the one of the
binary 7869181 (that node refuses the new payload: `MalformedObject`): every node of a rehearsal runs the same
build; a rehearsal is redeployed as a whole, never node by node.
