# Build from source

The workspace is Rust 2021 and pins Rust `1.96.0`. Native dependencies are
needed for MDBX, proof code and GUI packaging.

## Host requirements

All platforms need:

- the pinned Rust toolchain with `rustfmt`;
- a native C/C++ compiler;
- CMake;
- libclang;
- Git.

On Debian or Ubuntu:

```sh
sudo apt update
sudo apt install --no-install-recommends \
  build-essential clang libclang-dev cmake pkg-config
```

The Linux GUI package additionally needs `appstreamcli` and `dpkg-deb`.
Windows release packaging uses Inno Setup 6. macOS packaging uses the standard
`codesign`, `iconutil` and `hdiutil` tools.

The repository's `rust-toolchain.toml` selects the compiler automatically:

```sh
rustup show active-toolchain
rustc --version
cargo --version
```

## Check the workspace

```sh
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
```

Build ordinary development binaries:

```sh
cargo build --locked \
  -p jetsam_node \
  -p jetsam-extminer \
  -p jetsam_gui \
  --bins
```

Development binaries exercise parsing, UI and non-production test paths. A
block-producing release requires the authenticated HistoryStep matrix pack
described below.

## Reproduce the soundness certificate

The production calculations and proof documents are in
[`jetsam_soundness`](https://github.com/jetsam-chain/jetsam/tree/master/jetsam_soundness).

```sh
cargo run --release --locked -p jetsam_soundness
cargo run --release --locked -p jetsam_soundness -- --exact
cargo test --release --locked -p jetsam_soundness
```

## Generate the proof pack

The canonical pack contains:

```text
v1/history-step.runtime
v1/history-step-c00.field-r1cs.zst
v1/history-step-c01.field-r1cs.zst
pins.env
SHA256SUMS
```

Generate B25 and B255 matrices from honest fixtures:

```sh
mkdir -p ../jetsam-artifacts
./scripts/generate_history_step_pack.sh \
  ../jetsam-artifacts/history-step-pack-v1 \
  --profile mainnet
```

Generation is expensive and only needs to be performed once for an unchanged
relation. Keep the pack outside `target/`.

`--profile` is required and has no default. The matrices freeze that profile's
development-fund addresses, because the payout constraint names them, and that
constraint is armed once per target-time day: a pack built for the wrong
network runs 959 blocks out of 960 and stops the chain dead on the 960th. The
finished pack records its profile and both addresses in `pins.env`, and a node
refuses to start on a pack that does not pin its own.

The script writes to a staging directory, derives semantic pins, authenticates
every artifact and publishes the completed directory atomically. It refuses to
overwrite an existing output path.

## Build native deliverables

Linux artifacts are built inside the pinned container, never on the host:

```sh
docker build -t jetsam-build:22.04 -f docker/release-linux.Dockerfile docker

docker run --rm \
  -v "$PWD":/src \
  -v "$PWD/../jetsam-artifacts":/packs:ro \
  -w /src jetsam-build:22.04 \
  ./scripts/build_release.sh --pack /packs/history-step-pack-v1
```

On macOS and Windows, run the script directly; there is no glibc to pin.

### Why Linux goes through a container

The container is not a convenience. `mdbx.c`, in the `mdbx-sys` crate, does
`#define _GNU_SOURCE` itself. In glibc 2.38 and later, `features.h` turns
`_GNU_SOURCE` into `_ISOC2X_SOURCE`, which makes `stdlib.h` redirect `strtol`
to `__isoc23_strtol@GLIBC_2.38`. The node then imports a 2.38 symbol and
refuses to start on Ubuntu 22.04, Debian 12 and Rocky 9 — the three bases most
miners rent. That binary shipped in v1.1.0 and has been produced twice since.

The guard is `_GNU_SOURCE`, not `__STDC_VERSION__`, so `-std=gnu17` and every
other compiler flag is inert by construction, and `-DMDBX_DISABLE_GNU_SOURCE=1`
does not compile. The only lever that works is compiling against glibc headers
older than 2.38.

Two gates enforce it rather than trusting this paragraph:

- `build_release.sh` refuses to start a Linux build on a host whose glibc is
  newer than 2.35, before spending the build time;
- after linking and before packaging, it runs `objdump -T` on `jetsam`,
  `jetsam-cli`, `jetsam-miner` and `jetsam-gui` and **fails** if any of them
  imports a symbol newer than `GLIBC_2.34`.

The second gate is the one that decides. The native smoke test cannot catch
this defect: it runs the binaries on the machine that built them, where they
work.

Add `--pack-v1-3 DIR` whenever the v1.3 fork is armed. A binary whose
`V1_3_ACTIVATION_HEIGHT` is set but which carries no v1.3 pack can verify no
block from that height on, and refuses to start rather than stop every node
running it at the same block.

The script:

1. authenticates the pack and derives its pins;
2. checks formatting and the complete workspace;
3. embeds runtime metadata and both matrices into the node;
4. builds Core, external miner and GUI;
5. runs the native release tests;
6. smoke-tests every executable;
7. checks the glibc floor of every Linux binary and fails above `GLIBC_2.34`;
8. packages the Core archive and native GUI installer;
9. verifies archive membership and writes SHA-256 sums.

Find the output:

```sh
cat target/release-builds/LAST_RELEASE
```

Use `--output PATH` to choose a fresh output directory. `--skip-tests` is for a
platform packaging job whose source revision has already passed the complete
release gates; it should not be used for an independent release build.

## Portable binaries

x86-64 releases are compiled against a portable process baseline. After
checking the host, runtime dispatch selects `pclmul`, `avx2+vpclmul` or
`avx512bw+vpclmul`. ARM64 selects `neon+pmull`.

Do not compile official artifacts with `target-cpu=native`. That would make the
binary depend on the build machine before runtime hardware checks can run.

## Reproducible archive details

On GNU tar hosts, `SOURCE_DATE_EPOCH` controls member timestamps and defaults
to zero. The Core archive has a fixed member set:

```text
README.txt
LICENSE
NOTICE
jetsam
jetsam-cli
jetsam-miner
```

The GUI package contains only the application and its private node. It does not
include operator CLI or external-mining tools.
