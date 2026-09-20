#!/usr/bin/env bash

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

SCRIPT_DIR="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=release_common.sh
source "$SCRIPT_DIR/release_common.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/generate_history_step_pack.sh OUTPUT_DIR --profile mainnet|testnet
                                                 [--generation v1|v1.3]

Generate one canonical two-class HistoryStep pack from honest genesis
fixtures, authenticate it, and publish it atomically at OUTPUT_DIR. Matrix
generation can take roughly 90 minutes on the reference laptop. OUTPUT_DIR
must not exist; keep it outside the disposable repository target/ directory.

--profile is required and has no default. The matrices freeze that profile's
fund addresses, because the development payout constraint names them, so a
pack built for the wrong network satisfies its rows on 959 blocks out of 960
and stops the chain dead on the 960th. The test chain learned this at block
1920 on 2026-09-17. The finished pack records its profile and both addresses
in pins.env, so it can be read back without being regenerated.

Without --generation this builds the launch pack, unchanged. A generation is
the relation the whole build runs under, not a label on the output, so a pack
belongs for ever to the one it was frozen under.
EOF
}

if (( $# < 1 )); then
  usage >&2
  exit 2
fi
if [[ $1 == -h || $1 == --help ]]; then
  usage
  exit 0
fi

OUTPUT_ARGUMENT=$1
shift
GENERATOR_GENERATION=()
while (( $# > 0 )); do
  case $1 in
    --generation)
      (( $# >= 2 )) || release_die "--generation needs a value"
      case $2 in
        v1|v1.3) ;;
        *) release_die "unknown pack generation: $2" ;;
      esac
      (( ${#GENERATOR_GENERATION[@]} == 0 )) || release_die "--generation given twice"
      GENERATOR_GENERATION=(--generation "$2")
      shift 2
      ;;
    --profile)
      # Which network the pack is for. The matrices freeze this profile's fund
      # addresses — the development payout constraint names them — so a pack
      # built for the wrong network satisfies its rows on 959 blocks out of 960
      # and stops the chain dead on the 960th. The test chain learned this at
      # block 1920 on 2026-09-17. There is no default: guessing is what cost
      # the test chain 1920 blocks.
      (( $# >= 2 )) || release_die "--profile needs a value"
      [[ -z $RELEASE_PACK_PROFILE ]] || release_die "--profile given twice"
      release_set_pack_profile "$2"
      shift 2
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n $RELEASE_PACK_PROFILE ]] || {
  usage >&2
  release_die "--profile is required (mainnet|testnet); this pack's matrices freeze that network's fund addresses"
}

OUTPUT_DIR=$(release_absolute_from_root "$OUTPUT_ARGUMENT")
OUTPUT_PARENT=$(dirname -- "$OUTPUT_DIR")
OUTPUT_NAME=$(basename -- "$OUTPUT_DIR")
[[ $OUTPUT_NAME != . && $OUTPUT_NAME != .. && -n $OUTPUT_NAME ]] || \
  release_die "invalid output directory: $OUTPUT_DIR"
mkdir -p -- "$OUTPUT_PARENT"
OUTPUT_PARENT=$(release_canonical_directory "$OUTPUT_PARENT")
OUTPUT_DIR="$OUTPUT_PARENT/$OUTPUT_NAME"
[[ ! -e $OUTPUT_DIR && ! -L $OUTPUT_DIR ]] || \
  release_die "output directory already exists: $OUTPUT_DIR"

STAGING_DIR="$OUTPUT_PARENT/.$OUTPUT_NAME.generating.$$"
[[ ! -e $STAGING_DIR && ! -L $STAGING_DIR ]] || \
  release_die "staging path already exists: $STAGING_DIR"
CURRENT_STAGE=initialization

on_error() {
  local status=$?
  if (( status != 0 )); then
    printf '\nFAILED during: %s\n' "$CURRENT_STAGE" >&2
    if [[ -e $STAGING_DIR ]]; then
      printf 'Partial generator output was preserved at: %s\n' "$STAGING_DIR" >&2
    fi
  fi
  exit "$status"
}
trap on_error EXIT

release_require_command cargo
release_require_command rustc
release_require_command sed

cd "$RELEASE_ROOT_DIR"
release_build_pack_tools 1

CURRENT_STAGE='canonical HistoryStep matrix generation'
printf '\n==> Generating the m22 and m24 class matrices at zstd level 19\n'
JETSAM_ARTIFACT_ZSTD_LEVEL=19 "$RELEASE_MATRIX_GENERATOR" "$STAGING_DIR" \
  ${GENERATOR_GENERATION[@]+"${GENERATOR_GENERATION[@]}"}

CURRENT_STAGE='pack authentication'
printf '\n==> Authenticating generated artifacts and deriving release pins\n'
release_validate_pack_layout "$STAGING_DIR" 0
release_compute_pack_pins "$STAGING_DIR"
release_write_pin_file "$STAGING_DIR"
release_write_sha256_manifest "$STAGING_DIR"
release_authenticate_pack "$STAGING_DIR" 1

CURRENT_STAGE='atomic pack publication'
mv -- "$STAGING_DIR" "$OUTPUT_DIR"

CURRENT_STAGE=complete
printf '\nSUCCESS\n'
printf '  canonical pack: %s\n' "$OUTPUT_DIR"
printf '  profile:        %s\n' "$RELEASE_PACK_PROFILE"
printf '  network fund:   %s\n' "$RELEASE_PACK_NETWORK_FUND_ADDRESS"
printf '  lab fund:       %s\n' "$RELEASE_PACK_LAB_FUND_ADDRESS"
printf '  pins:           %s\n' "$OUTPUT_DIR/pins.env"
printf '  checksums:      %s\n' "$OUTPUT_DIR/SHA256SUMS"
