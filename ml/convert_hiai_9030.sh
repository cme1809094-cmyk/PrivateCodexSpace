#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: bash ml/convert_hiai_9030.sh /path/to/tools_omg/omg /path/to/build" >&2
  exit 2
fi

project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ml="$project/ml"
omg="$(realpath "$1")"
mkdir -p "$2"
work="$(realpath "$2")"
cd "$work"

for target in 16 32 64 baseline; do
  length=$target
  options=()
  if [[ $target == baseline ]]; then
    length=64
    options+=(--all-final-queries)
  fi
  dir="$work/l$target"
  mkdir -p "$dir"
  python "$ml/rewrite_perf.py" "$ml/permission.onnx" "$dir/static.onnx" --length "$length" "${options[@]}"
  python "$ml/rewrite_projections.py" "$dir/static.onnx" "$dir/projections.onnx" --method conv
  python "$ml/rewrite_attention.py" "$dir/projections.onnx" "$dir/base.onnx" gemm norm maskreshape
  python "$ml/rewrite_attention_npu.py" "$dir/matmul.onnx" --source "$dir/base.onnx" --parts matmul
  python "$ml/mask_cast_first.py" "$dir/matmul.onnx" "$dir/permission.onnx"
  bash "$omg" --model="$dir/permission.onnx" --framework=5 --output="$dir/permission" \
    --platform=kirin9030 --target=omc '--input_type=input_ids:INT32;attention_mask:INT32' \
    --output_type=logits:FP32 > "$dir/compiler.log" 2>&1
  bash "$omg" --mode=1 --om="$dir/permission.omc" --json="$dir/permission.json" > "$dir/inspect.log" 2>&1
done

python "$ml/verify_performance.py" "$work" --platform kirin9030 --include-baseline --output "$work/verification.json"
for target in 16 32 64 baseline; do
  install -m 644 "$work/l$target/permission.omc" "$project/entry/src/main/resources/rawfile/permission_$target.om"
done
