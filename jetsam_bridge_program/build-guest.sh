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
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
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
export RUSTC=${SP1_RUSTC:-/opt/jetsam-proto-zkvm/sp1/toolchain/bin/rustc} RUSTC_BOOTSTRAP=1 CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$out/target-$kind-$profile}
flags=$'-C\x1fpasses=lower-atomic\x1f-C\x1flink-arg=--image-base=2013265920\x1f-C\x1fpanic=abort\x1f--cfg\x1fgetrandom_backend="custom"\x1f-C\x1fllvm-args=-misched-prera-direction=bottomup\x1f-C\x1fllvm-args=-misched-postra-direction=bottomup'
mkdir -p "$out"
(cd "$here/$dir" && CARGO_ENCODED_RUSTFLAGS="$flags" \
  cargo +1.96.0 build --release --offline --locked --target riscv64im-succinct-zkvm-elf -j "${JOBS:-8}" "${feat[@]}")
cp "$CARGO_TARGET_DIR/riscv64im-succinct-zkvm-elf/release/$pkg" "$out/$kind-v3-$profile.elf"
sha256sum "$out/$kind-v3-$profile.elf"
echo "GUEST BUILD OK ($kind $profile)"
