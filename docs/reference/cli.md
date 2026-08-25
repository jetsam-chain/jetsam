# Command-line interfaces

The Core archive contains three executables:

| Executable | Role |
|---|---|
| `elide` | Full node, wallet backend and optional internal proof/mining pipeline |
| `elide-cli` | Thin JSON-RPC client for a running node |
| `elide-miner` | Separate Poseidon2b nonce-search worker |

## Core daemon

Run an ordinary node with no mode argument:

```sh
elide
```

Public daemon options are:

| Option | Meaning |
|---|---|
| `-c, --config FILE` | TOML configuration path |
| `--mode node` | Ordinary non-mining node |
| `--mode miner` | Node with internal proof and PoW pipeline |
| `--mode extminer` | Node-owned proof pipeline with external nonce worker |
| `--miner` | Shorthand for `--mode miner` |
| `--extminer` | Shorthand for `--mode extminer` |
| `--miner-address ADDRESS` | Mining payout address; active wallet owner is the default |
| `--cpu-threads N` | Shared internal-miner thread budget |
| `--data-dir PATH` | Chain, wallet and runtime data directory |
| `--p2p-listen HOST:PORT` | P2P listener; default `0.0.0.0:9600` |
| `--rpc-listen HOST:PORT` | JSON-RPC listener; default `127.0.0.1:9601` |
| `--seed ENDPOINT` | Add a bootstrap endpoint; repeatable |
| `--log LEVEL` | Tracing filter such as `error`, `warn`, `info` or `debug` |
| `--mining-key TOKEN` | Require a bearer token on RPC |
| `--allow-custom-coinbase` | Permit an authenticated external worker to request its payout |
| `--purge-state` | Clear the complete chain database and synchronize it again from peers |
| `--check-hardware` | Report production CPU support and exit without touching node data |

Mode-specific options are rejected when they cannot apply. `--cpu-threads`
belongs to internal miner mode. External-miner mode requires
`--mining-key`; `--allow-custom-coinbase` requires both external-miner mode
and that key.

`--purge-state` is a repair and upgrade tool, not a routine start option. It
removes headers, chain indexes, retained blocks, undo data and Live State from
the chain database. Wallet files, receipts and peer identity are stored
separately and remain; the node then authenticates the chain and State again
from peers.

Examples:

```sh
# Ordinary node
elide --data-dir ~/.elide/data

# Internal miner using 12 logical CPUs
elide --mode miner --cpu-threads 12

# Node with a local external nonce worker
elide --mode extminer --mining-key 'LONG-RANDOM-TOKEN'
```

Keep RPC on loopback unless it is protected by a private or authenticated
transport. A bearer key authenticates requests but does not encrypt them.

## External miner

`elide-miner` requests an already-proved, immutable template and searches
only its 128-bit nonce.

| Option | Default | Meaning |
|---|---|---|
| `--rpc URL` | `http://127.0.0.1:9601` | Node JSON-RPC endpoint |
| `--key TOKEN` | — | Bearer token matching the node's `--mining-key` |
| `--threads N` | `0` | PoW threads; zero uses every visible logical CPU |
| `--coinbase ADDRESS` | — | Custom `o1…` payout when the node explicitly permits it |
| `--poll-ms MS` | `500` | Delay before requesting another template |
| `--log LEVEL` | `info` | Worker log filter |
| `--check-hardware` | — | Report production CPU support and exit |

Typical local use:

```sh
elide-miner \
  --rpc http://127.0.0.1:9601 \
  --key 'LONG-RANDOM-TOKEN' \
  --threads 12
```

See [External miner](../operate/external-miner.md) for the remote transport and
custom-payout boundaries.

## Node client

`elide-cli` is a thin client for a running Core node. It holds no separate key
and performs wallet operations through local JSON-RPC.

### Global options

```text
-r, --rpc URL   RPC endpoint
-j, --json      raw JSON output
```

The default endpoint is `http://127.0.0.1:9601`. Environment variable
`ELIDE_RPC` changes it. `--rpc` takes precedence.

Amounts entered by CLI wallet commands are in ELD with up to six decimal
places:

```text
1 ELD = 1,000,000 μNOID
```

### Node and chain

| Command | Arguments | Result |
|---|---|---|
| `status` | — | Tip, hash, difficulty and active State summary |
| `block-hash` | `HEIGHT` | Permanent nonce-bearing block ID |
| `block-header` | `HEIGHT` | Structured permanent header |
| `header` | `HEIGHT` | Raw canonical 212-byte header hex |
| `block` | `HEIGHT` | Raw full block hex when retained |
| `history-step-terminal` | — | Current local terminal hex |
| `tx` | `TXID` | Permanent confirmation location |
| `slot` | `INDEX` | Current slot contents |
| `utxos-of` | `o1…` | Current UTXOs owned by an address |
| `state` | — | Capacity, occupancy and encoded State size |
| `epoch` | — | Current 144-block user transaction anchor |

Examples:

```sh
elide-cli status
elide-cli block-header 420
elide-cli block 420
elide-cli slot 9700063
elide-cli utxos-of o1...
```

`block` returns data only inside the 18-block body-retention window. `header`
and `block-header` remain available for every canonical height.

### Network and mining

| Command | Options | Result |
|---|---|---|
| `peers` | — | Connected peer count |
| `mining` | — | Difficulty, next reward and mining state |
| `block-template` | `--miner-addr o1…` | Node-owned external mining template |
| `submit-block` | `TEMPLATE_ID NONCE_HEX` | Submit one 16-byte little-endian nonce |

The external-mining commands are low-level. `elide-miner` handles template
refresh and nonce encoding automatically.

### Fees and mempool

```sh
elide-cli estimate-fee 2 --inputs 1
elide-cli mempool
elide-cli mempool-tx TXID
```

`estimate-fee` reports the current node-accepted minimum for the declared live
input and output counts. Mempool rows are logical `PagedSpend` intents, not
individual physical pages.

### Address validation

```sh
elide-cli validate o1...
```

The command reports validity, canonical bech32m form and raw 32-byte payload.

### Wallet addresses

```sh
elide-cli address
elide-cli address --list
elide-cli address --new
elide-cli address --index 3
elide-cli address --use 3
```

`--new`, `--index` and `--use` are mutually exclusive actions. A generated
address is inactive until selected. Sends use only the active owner's UTXOs
and return change to that owner.

### Balance and UTXOs

```sh
elide-cli balance
elide-cli utxos
elide-cli scan
```

`balance` separates confirmed, reserved outbound, pending incoming and
spendable values. `scan` reloads the active owner from the verified persistent
owner index.

### Send

Preview:

```sh
elide-cli send o1... 10.5 --dry-run
```

Submit with automatic fee:

```sh
elide-cli send o1... 10.5
```

Specify an exact fee in ELD only when needed:

```sh
elide-cli send o1... 10.5 --fee 0.012
```

The returned ID names the complete logical spend. Automatic fee selection is
recommended because occupancy and relay floor can change.

### History and receipts

```sh
elide-cli history
elide-cli history --last 20
elide-cli history --address o1...
```

Export a confirmed outgoing receipt:

```sh
elide-cli receipt TXID > receipt.hex
```

Verify it:

```sh
elide-cli verify "$(tr -d '\n' < receipt.hex)"
```

Receipt export works only for a locally saved different-owner payment.

### Stop

```sh
elide-cli stop
```

This requests graceful daemon shutdown. It is equivalent to the GUI's normal
exit path, not a process kill.

### Scripting

Use `--json` and check the process exit status:

```sh
height="$(
  elide-cli --json status |
    jq -er '.height'
)"
```

Human-readable output and colors are presentation interfaces. Scripts should
consume JSON.
