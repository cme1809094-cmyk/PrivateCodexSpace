#!/usr/bin/env bash
set -euo pipefail

task_tools_root="${1:-${HARMONY_BUILD_ROOT:-}}"
if [[ -z "$task_tools_root" || ! -f "$task_tools_root/activate.sh" ]]; then
  echo "Run python3 tools/harmony/setup.py --root /absolute/toolchain/path first." >&2
  exit 2
fi
source "$task_tools_root/activate.sh"
task_project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$task_project_root"
task_logs="${HARMONY_BUILD_LOGS:-$HARMONY_BUILD_ROOT/logs}"
mkdir -p "$task_logs"
{
  java -version
  node --version
  ohpm --version
  hvigorw --version
} > "$task_logs/environment.log" 2>&1
python3 ml/verify_artifacts.py | tee "$task_logs/artifacts.log"
ohpm install --all 2>&1 | tee "$task_logs/ohpm-install.log"
hvigorw clean --mode module -p product=default -p module=entry@default -p buildMode=debug \
  --no-daemon --stacktrace 2>&1 | tee "$task_logs/clean.log"
hvigorw assembleHap --mode module -p product=default -p module=entry@default -p buildMode=debug \
  --no-daemon --stacktrace 2>&1 | tee "$task_logs/assemble-hap.log"
python3 tools/harmony/verify_hap.py \
  entry/build/default/outputs/default/entry-default-unsigned.hap \
  --output "$task_logs/hap-verification.json"
