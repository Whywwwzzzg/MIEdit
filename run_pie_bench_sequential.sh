#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/haiyan/miniconda3_packed/bin/python}"
GPU="${GPU:-7}"
RESULT_ROOT="${RESULT_ROOT:-/data/disk2/haiyan/camera_ready_results}"
RUN_NAME="sd35_maxk11_maxv12_cfgdiff_8_13_MIEdit_SD3_5_sequential"

cd "$SCRIPT_DIR"

# One process is required to preserve the original process-level CUDA RNG stream.
exec env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u run_pie_bench.py \
    --target_path "$RESULT_ROOT/$RUN_NAME" \
    --eval_result_path "$RESULT_ROOT/eval_${RUN_NAME}.csv" \
    --eval_summary_path "$RESULT_ROOT/eval_${RUN_NAME}_summary.csv"
