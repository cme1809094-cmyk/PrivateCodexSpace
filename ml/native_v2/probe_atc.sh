#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  echo "Usage: bash probe_atc.sh ATC_EXT_BIN OUT" >&2
  exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
atc=$(realpath "$1")
mkdir -p "$2"
out=$(mktemp -d "$(realpath "$2")/atc.XXXXXX")
printf 'ATC logs and outputs in %s\n' "$out"

# Both CANN Python initialization and the mobile compiler need OS facilities
# that may be unavailable in a restricted build environment.
"${PYTHON_BIN:-python3}" - <<'PY' > "$out/host.log" 2>&1 || true
import ctypes
import socket

libc = ctypes.CDLL(None, use_errno=True)
addresses = ctypes.c_void_p()
result = libc.getifaddrs(ctypes.byref(addresses))
print(f"getifaddrs: rc={result} errno={ctypes.get_errno()}")
if result == 0:
    libc.freeifaddrs(addresses)
try:
    with socket.socket(socket.AF_UNIX):
        print("AF_UNIX socket: ok")
except OSError as error:
    print(f"AF_UNIX socket: {error}")
PY

for name in add qk_64_64 pv_64_64 softmax_64_64 qk_64_1 pv_64_1 softmax_64_1; do
  mkdir "$out/$name"
  if "$atc" --singleop="$here/specs/$name.json" --output="$out/$name" \
       --soc_version=Kirin9030 --log=info > "$out/$name/compiler.log" 2>&1 &&
       compgen -G "$out/$name/*.om" >/dev/null; then
    printf '%s: compiled single-op model\n' "$name"
  else
    printf '%s: compilation failed. See %s/%s/compiler.log\n' "$name" "$out" "$name" >&2
    exit 1
  fi
done
