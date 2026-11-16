#!/usr/bin/env bash
# Caller initializes the Mobile CANN and Ascend C environments.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash inspect_mobile.sh MOBILE_DDK_ROOT BUILD" >&2
  exit 2
fi
export DDK_PATH=$(realpath "$1")
build=$(realpath "$2")
master="$DDK_PATH/tools/tools_omg/master"
platform="$DDK_PATH/tools/platform/kirin9030"
export SOC_VERSION=kirin9030 CCE_LIB_DIR="$master/lib64/"
export LD_LIBRARY_PATH="$master/lib64:$platform/lib64:$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
for length in 16 32 64; do
  dir="$build/l$length"
  "${OMG_LOADER:-/lib64/ld-linux-x86-64.so.2}" "$master/omg" \
    --mode=1 --om="$dir/mobile.omc" --json="$dir/mobile.json" > "$dir/inspect-mobile.log" 2>&1
  test -s "$dir/mobile.json"
done
