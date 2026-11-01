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
