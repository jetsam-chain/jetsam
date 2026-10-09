# External miner

External mining separates PoW nonce search from the node. The node still owns
the mempool, transaction selection, State transition, `HistoryStep` proof,
template and block relay.

The worker receives no block body or proving witness.

## Local worker

Start the node in external-miner mode with a bearer token:

```sh
jetsam --mode extminer --mining-key 'LONG-RANDOM-TOKEN'
```

In another terminal:

```sh
jetsam-miner \
  --rpc http://127.0.0.1:9701 \
  --key 'LONG-RANDOM-TOKEN'
```

The token is required even on loopback when the node was started with
`--mining-key`.

Limit worker threads when needed:

```sh
jetsam-miner --key 'LONG-RANDOM-TOKEN' --threads 8
```

Each walking thread keeps a 512 KiB scratchpad in the core's L2 cache, so one
thread per physical core is usually the best setting. Measure first with
`jetsam --bench` and read its `walked digest` line.

## Remote worker

Do not expose an unencrypted bearer token and general RPC interface directly
to the Internet.

Place the worker and node on an authenticated private network, or terminate TLS
and restrict the exposed path at a reverse proxy. Bind public RPC only after
that transport is in place:

```sh
jetsam \
  --mode extminer \
  --rpc-listen 0.0.0.0:9701 \
  --mining-key 'LONG-RANDOM-TOKEN'
```

Firewall the port so only intended workers or the proxy can reach it.

## Payout

By default, templates use the node's configured payout address. This is the
safer solo-mining mode.

To let a worker request its own payout, the node operator must opt in:

```sh
jetsam \
  --mode extminer \
  --mining-key 'LONG-RANDOM-TOKEN' \
  --allow-custom-coinbase
```

The worker can then use:

```sh
jetsam-miner \
  --key 'LONG-RANDOM-TOKEN' \
  --coinbase j1...
```

Custom coinbase changes only the payout embedded before proof construction.
The worker still cannot modify the proved template.

## Template lifecycle

`getBlockTemplate` returns an opaque single-use ID, 16-field PoW schedule,
nonce index, target and, from block 24,846, `pow_walk: true`. With that flag
the worker must search `TowerWalk(TowerHash(fields))` instead of the TowerHash
digest alone; `jetsam-miner` reads the flag, never the height. The worker
searches random, independent nonce ranges and calls `submitBlock` with exactly
16 little-endian nonce bytes.

Each request from `jetsam-miner` carries two HTTP headers: `X-Jetsam-Version`
(its release) and `X-Jetsam-PoW: walk` (it can search the walked digest). A
node ignores both. A pool uses the second to tell walking miners from pre-fork
miners, which cannot find a block on TowerWalk work.

A template expires after 120 seconds. It is also invalidated by a canonical tip
change, successful submission or node-side cancellation. A stale result is
normal and the worker requests another template after its poll interval.

## Diagnose

Run:

```sh
jetsam-miner --check-hardware
```

If requests fail:

- `401 Unauthorized` means the token is absent or does not match;
- a custom coinbase error means the node did not enable it;
- repeated stale templates usually mean the node is receiving new tips or
  proof preparation exceeds the template lifecycle;
- `-32025` (digest not below the target) on every submission means the worker
  ignored `pow_walk` — a miner built before v1.4 searches the wrong digest;
- no template means the node is not synchronized, lacks the peer quorum or is
  not in `extminer` mode.
