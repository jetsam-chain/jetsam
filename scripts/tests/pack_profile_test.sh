#!/usr/bin/env bash

# Release-tooling guards around the pack profile.
#
# A pack freezes the fund addresses of the profile its generator was compiled
# under: the development payout constraint names them, so a pack built for the
# wrong network satisfies its rows on 959 blocks out of 960 and stops the chain
# dead on the 960th. The test chain learned this at block 1920 on 2026-09-17.
#
# These are the three script-level guards that stop it being repeated:
#
#  1. the profile is declared explicitly, never defaulted;
#  2. it comes from the command line, never from the environment;
#  3. the finished pack records it, so a pack can be read back without being
#     regenerated.
#
# Run: ./scripts/tests/pack_profile_test.sh

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(CDPATH='' cd -- "$SCRIPT_DIR/../.." && pwd -P)"
WORK_DIR="$(mktemp -d)"
FAILURES=0
CHECKS=0

cleanup() { rm -rf -- "$WORK_DIR"; }
trap cleanup EXIT

ok() {
  CHECKS=$(( CHECKS + 1 ))
  printf 'ok    %s\n' "$1"
}

fail() {
  CHECKS=$(( CHECKS + 1 ))
  FAILURES=$(( FAILURES + 1 ))
  printf 'FAIL  %s\n' "$1"
  if (( $# > 1 )); then
    printf '        %s\n' "${@:2}"
  fi
}

# Run a command, capturing status and merged output.
run() {
  RUN_OUTPUT=$("$@" 2>&1)
  RUN_STATUS=$?
  return 0
}

expect_failure_mentioning() {
  local label=$1 needle=$2
  shift 2
  set +e
  run "$@"
  set -e
  if (( RUN_STATUS == 0 )); then
    fail "$label" "expected a non-zero exit, got 0" "$RUN_OUTPUT"
  elif [[ $RUN_OUTPUT != *"$needle"* ]]; then
    fail "$label" "exit $RUN_STATUS but the message does not mention '$needle'" "$RUN_OUTPUT"
  else
    ok "$label"
  fi
}

# --- 1. --profile is required, and its absence is named ---------------------

expect_failure_mentioning \
  'generate_history_step_pack.sh refuses to guess a profile' \
  '--profile' \
  "$ROOT_DIR/scripts/generate_history_step_pack.sh" "$WORK_DIR/unused-pack"

expect_failure_mentioning \
  'generate_history_step_pack.sh refuses an unknown profile' \
  'unknown network profile' \
  "$ROOT_DIR/scripts/generate_history_step_pack.sh" "$WORK_DIR/unused-pack" \
  --profile devnet

# A refusal must not leave a half-made pack behind.
if [[ -e $WORK_DIR/unused-pack ]]; then
  fail 'a refused generation creates no output directory'
else
  ok 'a refused generation creates no output directory'
fi

# --- 2. the environment cannot decide the profile ---------------------------

# `release_common.sh` is sourced by every release script. Sourcing it with a
# profile-bearing variable already exported must leave nothing behind that a
# later build could read.
set +e
RUN_OUTPUT=$(
  JETSAM_PACK_TOOL_FEATURES=testnet bash -c '
    set -Eeuo pipefail
    source "$1/scripts/release_common.sh"
    printf "inherited=%s\n" "${JETSAM_PACK_TOOL_FEATURES:-<unset>}"
    printf "profile=%s\n" "${RELEASE_PACK_PROFILE:-<unset>}"
  ' _ "$ROOT_DIR" 2>&1
)
RUN_STATUS=$?
set -e
if (( RUN_STATUS != 0 )); then
  fail 'sourcing release_common.sh drops an inherited pack profile' \
    "sourcing failed with $RUN_STATUS" "$RUN_OUTPUT"
elif [[ $RUN_OUTPUT != *'inherited=<unset>'* ]]; then
  fail 'sourcing release_common.sh drops an inherited pack profile' \
    'JETSAM_PACK_TOOL_FEATURES survived into the build environment' "$RUN_OUTPUT"
elif [[ $RUN_OUTPUT != *'profile=<unset>'* ]]; then
  fail 'sourcing release_common.sh drops an inherited pack profile' \
    'RELEASE_PACK_PROFILE was inherited instead of declared' "$RUN_OUTPUT"
else
  ok 'sourcing release_common.sh drops an inherited pack profile'
fi

# An inherited RELEASE_PACK_PROFILE must not arm the tool build either.
set +e
RUN_OUTPUT=$(
  RELEASE_PACK_PROFILE=testnet bash -c '
    set -Eeuo pipefail
    source "$1/scripts/release_common.sh"
    release_build_pack_tools 0
  ' _ "$ROOT_DIR" 2>&1
)
RUN_STATUS=$?
set -e
if (( RUN_STATUS == 0 )); then
  fail 'release_build_pack_tools refuses an undeclared profile' \
    'the tool build ran with no declared profile' "$RUN_OUTPUT"
elif [[ $RUN_OUTPUT != *'profile'* ]]; then
  fail 'release_build_pack_tools refuses an undeclared profile' \
    "exit $RUN_STATUS but the message does not mention the profile" "$RUN_OUTPUT"
else
  ok 'release_build_pack_tools refuses an undeclared profile'
fi

# --- 3. the pack records its own identity -----------------------------------

MAINNET_NETWORK=77ca540bbf9e017b3e8a139fe91366bd00db8ca0518d113a4206664d29f044f0
MAINNET_LAB=71de51c5a9cfc4622c624783fb3cefb1d8104708a72c10191dc14bfc70147d8f
FAKE_METADATA_DIGEST=$(printf '9%.0s' {1..64})
FAKE_LEAF_DIGESTS=$(printf 'a%.0s' {1..128})

mkdir -p -- "$WORK_DIR/pack"
set +e
RUN_OUTPUT=$(
  bash -c '
    set -Eeuo pipefail
    source "$1/scripts/release_common.sh"
    release_set_pack_profile mainnet
    RELEASE_METADATA_DIGEST=$2
    RELEASE_LEAF_DIGESTS=$3
    RELEASE_PACK_NETWORK_FUND_ADDRESS=$4
    RELEASE_PACK_LAB_FUND_ADDRESS=$5
    release_write_pin_file "$6"
    release_read_pin_file "$6/pins.env"
    printf "profile=%s\n" "$RELEASE_FILE_PACK_PROFILE"
    printf "network=%s\n" "$RELEASE_FILE_NETWORK_FUND_ADDRESS"
    printf "lab=%s\n" "$RELEASE_FILE_LAB_FUND_ADDRESS"
    printf "metadata=%s\n" "$RELEASE_FILE_METADATA_DIGEST"
  ' _ "$ROOT_DIR" "$FAKE_METADATA_DIGEST" "$FAKE_LEAF_DIGESTS" \
    "$MAINNET_NETWORK" "$MAINNET_LAB" "$WORK_DIR/pack" 2>&1
)
RUN_STATUS=$?
set -e
if (( RUN_STATUS != 0 )); then
  fail 'pins.env carries the profile and both fund addresses' \
    "round trip failed with $RUN_STATUS" "$RUN_OUTPUT"
else
  missing=()
  [[ $RUN_OUTPUT == *"profile=mainnet"* ]] || missing+=('profile')
  [[ $RUN_OUTPUT == *"network=$MAINNET_NETWORK"* ]] || missing+=('network fund')
  [[ $RUN_OUTPUT == *"lab=$MAINNET_LAB"* ]] || missing+=('lab fund')
  [[ $RUN_OUTPUT == *"metadata=$FAKE_METADATA_DIGEST"* ]] || missing+=('metadata digest')
  if (( ${#missing[@]} > 0 )); then
    fail 'pins.env carries the profile and both fund addresses' \
      "not read back: ${missing[*]}" "$RUN_OUTPUT"
  else
    ok 'pins.env carries the profile and both fund addresses'
  fi
fi

# A pack published before pins.env carried an identity still reads: the two
# legacy assignments are a complete file, with an empty declared profile.
printf 'JETSAM_HISTORY_STEP_RUNTIME_METADATA_RELEASE_DIGEST=%s\n' "$FAKE_METADATA_DIGEST" \
  > "$WORK_DIR/pack/pins.env"
printf 'JETSAM_HISTORY_STEP_PACK_LEAF_DIGESTS=%s\n' "$FAKE_LEAF_DIGESTS" \
  >> "$WORK_DIR/pack/pins.env"
set +e
RUN_OUTPUT=$(
  bash -c '
    set -Eeuo pipefail
    source "$1/scripts/release_common.sh"
    release_read_pin_file "$2/pins.env"
    printf "profile=[%s]\n" "$RELEASE_FILE_PACK_PROFILE"
  ' _ "$ROOT_DIR" "$WORK_DIR/pack" 2>&1
)
RUN_STATUS=$?
set -e
if (( RUN_STATUS != 0 )); then
  fail 'a pack published before the identity lines still reads' \
    "read failed with $RUN_STATUS" "$RUN_OUTPUT"
elif [[ $RUN_OUTPUT != *'profile=[]'* ]]; then
  fail 'a pack published before the identity lines still reads' \
    'an absent profile should read back empty' "$RUN_OUTPUT"
else
  ok 'a pack published before the identity lines still reads'
fi

# --- 4. build_release.sh knows two packs ------------------------------------

set +e
RUN_OUTPUT=$("$ROOT_DIR/scripts/build_release.sh" --help 2>&1)
RUN_STATUS=$?
set -e
if (( RUN_STATUS != 0 )); then
  fail 'build_release.sh documents a second pack' "--help exited $RUN_STATUS" "$RUN_OUTPUT"
elif [[ $RUN_OUTPUT != *'--pack-v1-3'* ]]; then
  fail 'build_release.sh documents a second pack' \
    'no --pack-v1-3 option' "$RUN_OUTPUT"
else
  ok 'build_release.sh documents a second pack'
fi

expect_failure_mentioning \
  'build_release.sh names a missing second pack directory' \
  '--pack-v1-3 requires a directory' \
  "$ROOT_DIR/scripts/build_release.sh" --pack "$WORK_DIR/pack" --pack-v1-3

printf '\n%d checks, %d failed\n' "$CHECKS" "$FAILURES"
(( FAILURES == 0 ))
