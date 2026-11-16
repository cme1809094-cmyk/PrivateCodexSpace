#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash ml/convert.sh /path/to/mindspore-lite-2.1.0-linux-x64 [permission.onnx]" >&2
  exit 2
fi

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
lite="$(cd -- "$1" && pwd)"
onnx_model="${2:-$project/model-build/permission.onnx}"
output="$project/entry/src/main/resources/rawfile/permission"
export LD_LIBRARY_PATH="$lite/tools/converter/lib:$lite/runtime/lib:$lite/runtime/third_party/glog${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$lite/tools/converter/converter/converter_lite" \
  --fmk=ONNX \
  --modelFile="$onnx_model" \
  --outputFile="$output" \
  --inputShape="input_ids:1,64;attention_mask:1,64"
