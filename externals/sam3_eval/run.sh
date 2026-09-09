#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config.yaml}"
GPU_COUNT="${2:-1}"

source /inspire/hdd/global_user/chenxinyan-240108120066/miniconda3/bin/activate ggearth

export PYTHONPATH=/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/sam3-eval/sam3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$GPU_COUNT" -le 1 ]]; then
  python run_eval.py --config "$CONFIG_PATH"
else
  torchrun \
    --standalone \
    --nproc_per_node "$GPU_COUNT" \
    run_eval.py \
    --config "$CONFIG_PATH"
fi
