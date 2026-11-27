#!/usr/bin/env bash
# Reproducible release build of the guardian programs (T5.5, design §7.1).
#   reproducible-build.sh <source dir (a git archive of the commit)> <target dir>
# Prints the SHA-256 of jg-observe, jg-net, jg-sign, jg-aggregate. Two runs (two hosts, or two source
# directories) must print the same four lines.
set -Eeuo pipefail
src=$(cd "${1:?source dir}" && pwd); tgt=${2:?target dir}
export PATH="$HOME/.cargo/bin:$PATH" CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=$tgt
export SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-1793404800}
reg=$(cd "$HOME/.cargo/registry" && pwd)
export RUSTFLAGS="--remap-path-prefix=$src=/build/jetsam --remap-path-prefix=$reg=/build/registry --remap-path-prefix=$HOME=/build/home -C debuginfo=0"
cd "$src"
nice -n 10 cargo build --locked --offline --release -j "${JOBS:-24}" -p jetsam_guardian \
  --bin jg-observe --bin jg-net --bin jg-sign --bin jg-aggregate >&2
cd "$tgt/release" && sha256sum jg-observe jg-net jg-sign jg-aggregate
