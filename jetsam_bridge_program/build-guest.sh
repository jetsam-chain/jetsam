#!/usr/bin/env bash
# Builds a bridge program guest (v3) for the SP1 6.10.0 zkVM with the succinct 1.96.0-64bit
# toolchain and exactly sp1-build 6.10.0's flags (as the v2 prototype did), offline.
#   build-guest.sh <claim|checkpoint> <mainnet|testnet|rehearsal|test-clock> <out-dir>
#     claim      -> <out-dir>/claim-v3-<profile>.elf       the deposit-claim program
#     checkpoint -> <out-dir>/checkpoint-v3-<profile>.elf  the checkpoint program
#   mainnet = the public network; testnet = the test chain; rehearsal = the private rehearsal clock
#   (J_R = 96); test-clock = TEST ONLY, the clock of the node's G6 = V4 vectors (J = 69 120).
# Environment: SP1_RUSTC   the succinct rustc (default /opt/jetsam-proto-zkvm/sp1/toolchain/bin/rustc)
#              CARGO_HOME  a cargo home whose registry holds sp1-zkvm 6.10.0 and its dependencies
#              CARGO_TARGET_DIR  (default <out-dir>/target-<kind>-<profile>)
#
# Reproducible (A5FIX5): the ELF, hence the vkey and the bridge's CREATE2 address, must not depend on
# where the tree or CARGO_HOME sit. Two things made it depend on them:
#   - CARGO_ENCODED_RUSTFLAGS REPLACES the repository's build.rustflags, and with them its
#     --remap-path-prefix: the absolute paths of the sources were written into the ELF (panic
#     locations). The remap is therefore given here, with the guest's flags.
#   - cargo derives the crate ids (-C metadata) of path dependencies that lie outside the building
#     workspace (core, towerwalk, jetsam_vault_rule: the guest is its own workspace) from their
#     ABSOLUTE path (measured, A5FIX5: only those crate ids differ between two trees), and with them
#     the string constants are laid out in another order, so the code addresses and the vkey moved
#     with the directory (presumably the order in which fat LTO merges the modules). No remap
#     reaches them.
# So the build runs in a private mount namespace in which this tree is always FIXED/jetsam and
# CARGO_HOME always FIXED/cargo (the way SP1's own reproducible builds use a container with fixed
# paths): every path cargo and rustc see is the same on every machine. Nothing is mounted outside
# the namespace; root, or unprivileged user namespaces (--map-root-user), are required.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd -P)
root=$(cd "$here/.." && pwd -P)
kind=${1:?claim|checkpoint}; profile=${2:?mainnet|testnet|rehearsal|test-clock}; out=${3:?out-dir}
case $kind in
  claim) dir=guest-claim; pkg=jetsam-bridge-claim-guest;;
  checkpoint) dir=guest-checkpoint; pkg=jetsam-bridge-checkpoint-guest;;
  *) echo "kind: claim or checkpoint" >&2; exit 2;;
esac
case $profile in
  mainnet) feat=();;
  testnet) feat=(--features testnet);;
  rehearsal) feat=(--features testnet,rehearsal-v1-7);;
  test-clock) feat=(--features test-clock-j69120);;
  *) echo "profile: mainnet, testnet, rehearsal or test-clock" >&2; exit 2;;
esac
: "${CARGO_HOME:?CARGO_HOME with an offline registry holding sp1-zkvm 6.10.0}"
cargo_home=$(cd "$CARGO_HOME" && pwd -P)
mkdir -p "$out"; out=$(cd "$out" && pwd -P)
export RUSTC=${SP1_RUSTC:-/opt/jetsam-proto-zkvm/sp1/toolchain/bin/rustc} RUSTC_BOOTSTRAP=1 CARGO_INCREMENTAL=0
CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$out/target-$kind-$profile}
mkdir -p "$CARGO_TARGET_DIR"; CARGO_TARGET_DIR=$(cd "$CARGO_TARGET_DIR" && pwd -P); export CARGO_TARGET_DIR
fixed=/tmp/jetsam-guest-build
mkdir -p "$fixed/jetsam" "$fixed/cargo"
flags="--remap-path-prefix=$fixed/jetsam=/jetsam"$'\x1f'"--remap-path-prefix=$fixed/cargo=/cargo"$'\x1f-C\x1fpasses=lower-atomic\x1f-C\x1flink-arg=--image-base=2013265920\x1f-C\x1fpanic=abort\x1f--cfg\x1fgetrandom_backend="custom"\x1f-C\x1fllvm-args=-misched-prera-direction=bottomup\x1f-C\x1fllvm-args=-misched-postra-direction=bottomup'
userns=(); [ "$(id -u)" = 0 ] || userns=(--map-root-user)
unshare --mount --propagation private "${userns[@]}" -- bash -c '
  set -euo pipefail
  fixed=$1 root=$2 cargo_home=$3 dir=$4 flags=$5; shift 5
  mount --bind "$root" "$fixed/jetsam"
  mount --bind "$cargo_home" "$fixed/cargo"
  cd "$fixed/jetsam/jetsam_bridge_program/$dir"
  CARGO_HOME="$fixed/cargo" CARGO_ENCODED_RUSTFLAGS="$flags" \
    cargo +1.96.0 build --release --offline --locked --target riscv64im-succinct-zkvm-elf -j "${JOBS:-8}" "$@"
' build-guest "$fixed" "$root" "$cargo_home" "$dir" "$flags" "${feat[@]}"
cp "$CARGO_TARGET_DIR/riscv64im-succinct-zkvm-elf/release/$pkg" "$out/$kind-v3-$profile.elf"
sha256sum "$out/$kind-v3-$profile.elf"
echo "GUEST BUILD OK ($kind $profile)"
