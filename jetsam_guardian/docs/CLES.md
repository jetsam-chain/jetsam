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
- **Process** (A5-15): `jg-fill` and `jg-relay` harden themselves before reading anything
  (`jetsam_guardian::harden`): `PR_SET_DUMPABLE=0` and `RLIMIT_CORE=0` (they do not start otherwise);
  `mlockall` when `RLIMIT_MEMLOCK` is unlimited or the process has `CAP_IPC_LOCK`, else a warning — then the
  host's swap must be off or encrypted. Started by `ops/jg-start fill|relay` only (SHA-256 of the binary and of
  the configuration in `allowed.sha256`, one instance, PID file); `jg-fill` also holds a `flock` on its
  `state_dir` and refuses a `fill.toml` that is not 0600/0400 and owned by its uid (it names `refund_to`).
- **Stock** (robot): wJTM balance of the hot key ≤ `max_stock` (1 200 JTM by default), checked at start only
  (a refill while it runs is used as is: a question noted, not decided); the contract adds 3 000 JTM
  outstanding and 2 500 JTM / 24 h per advancer. **The maximum loss per incident is the hot key's stock.**
  Repayments go to `refund_to`, a cold address other than the hot key.
- **Fee of an advance** (A5-1, contract revised: `ADVANCE_MIN_AGE` = `D_SLOW` = 24 h): an advance less than 24 h
  old at the mint of its deposit is repaid what it paid only, its fee going to the recipient. An advance made before
  the acceptance of the point that covers its deposit is always 24 h old at the mint; so `jg-fill` never advances a
  deposit at or below the locked height of the contract's tip (`ALREADY_UNDER_A_POINT`: the slow path pays the
  recipient within 24 h), and does not start unless the bridge's `ADVANCE_MIN_AGE()` is 86 400. **An advance
  doubled by the relayer** (its point covering the deposit lands in the few minutes between the robot's read of the
  tip and its advance) **loses its fee: accepted** (the recipient gets it; the robot is repaid what it paid).
- Holder of each hot key and rotation procedure: **[NV]** (to be written here).

## Loss per incident (decision 11 of the audit A5)

What is said publicly: **the maximum loss per incident is the robot's stock (`max_stock`)**. The contract's
per-advancer outstanding cap (`ADVANCE_OUTSTANDING_CAP`, 3 000 JTM) is a circuit breaker against a faulty
robot, not a property of the system; no figure is derived from a pool's dated share. The bridge assumes an
honest majority of the mining (a majority can prove deposits that do not exist, A5-21).

## Relayers (decision 12 of the audit A5, A5-9)

A second relayer is allowed only with a contract that carries the A5-4 fix (`MustExtendTip`: a point on the
tip's own chain that does not extend the tip is refused, the finality clock does not move): before it, two
honest relayers out of step restarted the 24 h clock. Two relayers then share nothing but the L1: a proof
made while the other's point landed is void (refused, made again on the new tip).
