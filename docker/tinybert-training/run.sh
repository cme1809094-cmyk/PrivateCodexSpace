#!/usr/bin/env bash
set -euo pipefail

device=cuda
gpu_args=(--gpus all)
if [[ ${1:-} == --cpu ]]; then
    device=cpu
    gpu_args=()
    shift
fi

if (( $# < 2 )); then
    echo 'Usage: run.sh [--cpu] INPUTS_DIR RUNS_DIR [COMMAND ...]' >&2
    exit 2
fi

if [[ ! -d $1 ]]; then
    echo "Inputs directory does not exist: $1" >&2
    exit 2
fi
inputs=$(cd -- "$1" && pwd -P)
mkdir -p -- "$2"
runs=$(cd -- "$2" && pwd -P)
if [[ $inputs == "$runs" || $inputs == "${runs%/}/"* || $runs == "${inputs%/}/"* ]]; then
    echo 'Use separate input and output directories. Neither directory may contain the other.' >&2
    exit 2
fi
shift 2

exec docker run --rm --init "${gpu_args[@]}" \
    --user "$(id -u):$(id -g)" \
    --env "TINYBERT_TRAINING_DEVICE=$device" \
    --mount "type=bind,source=$inputs,target=/inputs,readonly" \
    --mount "type=bind,source=$runs,target=/runs" \
    "${TINYBERT_TRAINING_IMAGE:-tinybert-training:v4-cu128}" "$@"
