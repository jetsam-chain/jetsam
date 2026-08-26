# Internal mining

Internal mining keeps transaction selection, proof construction, nonce search
and block submission in one Core process.

## Prepare

Run the production hardware check:

```sh
jetsam --check-hardware
```

Obtain a payout address from the node wallet:

```sh
jetsam
jetsam-cli address --list
jetsam-cli stop
```

The active wallet address is used automatically when no explicit payout is
configured.

## Start

In the foreground:

```sh
jetsam --mode miner --cpu-threads 12
```

Or update a systemd unit:

```ini
ExecStart=/usr/local/bin/jetsam \
  --config /etc/jetsam/jetsam.toml \
  --mode miner \
  --cpu-threads 12
```

Then reload and restart:

```sh
sudo systemctl daemon-reload
sudo systemctl restart jetsam
```

## Readiness

Ordinary mining requires one authenticated peer and a synchronized chain.
Check:

```sh
jetsam-cli status
jetsam-cli peers
jetsam-cli mining
```

The process prepares its embedded B25 and B255 proof matrices and selects the
best available CPU backend. Every process starts block production with B25;
larger B255 templates are used only when measured complete preparation timing
supports them.

## CPU planning

`--cpu-threads` is the total shared budget for proof and PoW phases. Do not set
it higher than the logical CPUs available to the service's cgroup or virtual
machine.

Leave capacity for the operating system and public P2P service on an
infrastructure node. A dedicated miner can use every visible logical CPU.

Wallet transaction proving has local priority over ongoing mining work. This
does not change the transactions or blocks accepted by other nodes.

## Payout changes

The active wallet address is resolved for each new template. Changing it
invalidates or refreshes local work at a safe boundary. An already immutable
template cannot have its payout rewritten.

To pin a separate payout for the process:

```sh
jetsam --mode miner --miner-address o1...
```

Use the complete bech32m address.

## Stop

Stop through RPC or the service manager:

```sh
jetsam-cli stop
```

```sh
sudo systemctl stop jetsam
```

Graceful shutdown cancels mining, closes networking and flushes MDBX. Do not
send repeated hard-kill signals merely because an active proof takes several
seconds to reach its cancellation boundary.
