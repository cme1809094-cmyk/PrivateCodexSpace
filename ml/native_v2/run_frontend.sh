#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 || ( $# == 5 && $5 != --compile ) ]]; then
  echo "Usage: bash run_frontend.sh PROJECT CANN_INCLUDE MOBILE_DDK_ROOT OUT [--compile]" >&2
  exit 2
fi

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ml="$(realpath "$1")/ml"
include=$(realpath "$2")
export DDK_PATH=$(realpath "$3")
mkdir -p "$4"
out=$(mktemp -d "$(realpath "$4")/v2_XXXXXX")
master="$DDK_PATH/tools/tools_omg/master"
platform="$DDK_PATH/tools/platform/kirin9030"
loader=${OMG_LOADER:-/lib64/ld-linux-x86-64.so.2}
python_bin=${PYTHON_BIN:-python3}
printf 'Build directory %s\n' "$out"
"$python_bin" "$ml/inspect_bert.py" "$ml/permission.onnx" > "$out/architecture.json"

g++ -std=c++17 -shared -fPIC -I"$include" \
  "$here/plugin/onnx_native.cpp" "$here/plugin/onnx_prototypes.cpp" \
  -L"$master/lib64" -lai_register -lgraph_host -o "$out/libcustom_op.so" > "$out/plugin.log" 2>&1

export SOC_VERSION=kirin9030 CCE_LIB_DIR="$master/lib64/"
export LD_LIBRARY_PATH="$master/lib64:$platform/lib64:$out${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$platform/ops/impl${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$DDK_PATH/ddk/ccec_compiler/bin:$PATH"

for length in 16 32 64; do
  dir="$out/l$length"
  mkdir "$dir"
  cd "$dir"
  "$python_bin" "$ml/rewrite_perf.py" "$ml/permission.onnx" static.onnx --length "$length"
  "$python_bin" "$ml/rewrite_projections.py" static.onnx projections.onnx --method conv
  "$python_bin" "$ml/rewrite_attention.py" projections.onnx base.onnx gemm norm maskreshape
  # Cast the rank-2 integer mask before reshaping, as in the project's Kirin
  # pipeline. Reshaping the integer mask first creates a CPUCL subgraph.
  "$python_bin" "$ml/mask_cast_first.py" base.onnx mask.onnx
  norm_source=mask.onnx
  if [[ ${NATIVE_LAYERNORM_SCALE:-1} != 1 ]]; then
    "$python_bin" "$here/stabilize_layernorm.py" mask.onnx norm.onnx \
      --scale "$NATIVE_LAYERNORM_SCALE" --report norm-stability.json
    norm_source=norm.onnx
  fi
  "$python_bin" "$here/prepare_attention.py" "$norm_source" permission_native.onnx adapted.onnx

  args=(--model=adapted.onnx --framework=5 --platform=kirin9030
        --weight_merge=false '--input_type=input_ids:INT32;attention_mask:INT32'
        --output_type=logits:FP32)
  "$loader" "$master/omg" "${args[@]}" --target=om --output=frontend > frontend.log 2>&1
  "$loader" "$master/omg" --mode=1 --om=frontend.om --json=frontend.json > inspect.log 2>&1
  "$python_bin" "$here/verify_frontend.py" "$length" frontend.json --architecture "$out/architecture.json" > frontend-verified.json
  printf 'Length %s verified in %s/frontend-verified.json\n' "$length" "$dir"
done

if [[ ${5:-} == --compile ]]; then
  failed=0
  for length in 16 32 64; do
    dir="$out/l$length"
    cd "$dir"
    if "$loader" "$master/omg" --model=adapted.onnx --framework=5 --platform=kirin9030 \
         --target=omc --weight_merge=false '--input_type=input_ids:INT32;attention_mask:INT32' \
         --output_type=logits:FP32 --output=mobile > backend.log 2>&1 && [[ -s mobile.omc ]]; then
      printf 'OMC generated in %s/mobile.omc. Inspect V2 kernel assignments before device testing.\n' "$dir"
    else
      printf 'OMC generation failed at length %s. See %s/backend.log\n' "$length" "$dir" >&2
      failed=1
    fi
  done
  if [[ $failed != 0 ]]; then
    exit "$failed"
  fi
  bash "$here/inspect_mobile.sh" "$DDK_PATH" "$out"
  "$python_bin" "$here/verify_native_omc.py" "$out" --output "$out/native-verification.json"
fi
