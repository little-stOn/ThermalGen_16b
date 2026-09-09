#!/usr/bin/env bash
set -euo pipefail

cd /inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base
source .venv-butiv/bin/activate
export PYTHONPATH=src

export BUTIV_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/bu_tiv
export FLIR_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/flir
export LTIR_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/ltir_v1
export ZUT_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/zut_fir_adas
export MS2_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/ms2
export VIVID_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/vivid_pp
export LYNRED_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/lynred_mobility
export TARTAN_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/raw/tartanrgbt

export MS2_ANNOTATION_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/standardized/annotations/ms2_sam31_bbox_v1_highconf75
export VIVID_ANNOTATION_ROOT=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base/data/standardized/annotations/vivid_pp_driving_full_v4

TASK="${1:-joint}"
BATCHES="${BATCHES:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${TASK}_visual}"

case "${TASK}" in
  bbox|style|joint) ;;
  *)
    printf 'usage: bash start.sh [bbox|style|joint]\n' >&2
    exit 2
    ;;
esac

exec python scripts/prepare/visualize_dataset_batch.py \
  --task "${TASK}" \
  --batches "${BATCHES}" \
  --max-samples-per-batch 0 \
  --output-dir "${OUTPUT_DIR}" \
  --color-map inferno \
  --display-scale 2
