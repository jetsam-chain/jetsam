# Run Core locally

The Core archive is for node operators, miners and developers. It contains:

- `elide`, the full node and built-in miner;
- `elide-cli`, the local node and wallet client;
- `elide-miner`, the external PoW worker.

The GUI wallet is distributed separately.

## Download and verify

Choose the Core archive for the host:

```text
elide-core-vVERSION-linux-x86_64.tar.gz
elide-core-vVERSION-linux-aarch64.tar.gz
elide-core-vVERSION-windows-x86_64.zip
elide-core-vVERSION-macos-aarch64.tar.gz
elide-core-vVERSION-macos-x86_64.tar.gz
```

Compare its SHA-256 digest with `SHA256SUMS` from the same release before
extracting it.

## Check the CPU

On Linux or macOS:

```sh
./elide --check-hardware
```

On Windows PowerShell:

```powershell
.\elide.exe --check-hardware
```

The final line must be:

```text
NODE READY
```

Production requires SSE4.1 and PCLMULQDQ on x86-64, or NEON and PMULL on
ARM64. Runtime dispatch selects the `pclmul`, `avx2+vpclmul`,
`avx512bw+vpclmul` or `neon+pmull` backend automatically.

For virtual-machine caveats, memory and disk planning, see
[Hardware and capacity](../operate/hardware.md).

## Start an ordinary node

Run Core in the foreground:

```sh
./elide
```

On first start it creates:

```text
~/.elide/elide.toml
~/.elide/data/
```

In another terminal:

```sh
./elide-cli status
./elide-cli peers
./elide-cli state
```

The default P2P listener is `0.0.0.0:9600`. RPC remains on
`127.0.0.1:9601`.

Stop cleanly with:

```sh
./elide-cli stop
```

For a permanent system service, firewall guidance and updates, continue with
[Run a node on Linux](../operate/node.md). All commands are listed in the
[CLI reference](../reference/cli.md).
