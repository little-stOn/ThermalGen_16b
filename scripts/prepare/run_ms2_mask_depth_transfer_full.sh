#!/usr/bin/env bash
set -euo pipefail
P=/inspire/qb-ilm/project/wuliqifa/public/mayilin/ir_dwm_base
IN=$P/data/standardized/annotations/ms2_sam31_bbox_v1_highconf75
OUT=$P/data/standardized/evaluations/ms2_mask_depth_transfer_full_v0
PY=$P/.venv-butiv/bin/python
TOOL=$P/scripts/prepare/validate_ms2_mask_depth_transfer.py
SAM=$P/externals/sam3_eval
CKPT=$P/checkpoints/sam3.1/sam3.1_multiplex.pt
mkdir -p "$OUT/telemetry" "$OUT/shards"
nohup nvidia-smi --query-gpu=index,timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits --loop-ms=1000 > "$OUT/telemetry/gpu.csv" 2>&1 < /dev/null &
MONITOR_PID=$!
echo "monitor_pid=$MONITOR_PID"
pid_list=""
for rank in 0 1 2 3; do
  if [ "$rank" -lt 2 ]; then gpu=0; else gpu=1; fi
  shard=$(printf 'rank%02d' "$rank")
  mkdir -p "$OUT/shards/$shard"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" "$TOOL" --mode worker --project-root "$P" --input-root "$IN" --output-root "$OUT" --sam3-eval-root "$SAM" --checkpoint "$CKPT" --frames-per-sequence 0 --max-boxes-per-frame 0 --preview-count 5 --preview-stride 10000 --batch-size 16 --rank "$rank" --world-size 4 --device cuda:0 --input-resolution 1008 --mask-resolution 256 --model-confidence 0.75 --nms-iou 0.7 --mask-threshold 0.5 --precision bfloat16 --final-only > "$OUT/telemetry/$shard.log" 2>&1 < /dev/null &
  pid=$!
  pid_list="$pid_list $pid"
  echo "$shard pid=$pid gpu=$gpu"
done
status=0
for pid in $pid_list; do
  if ! wait "$pid"; then status=1; fi
done
if [ "$status" -ne 0 ]; then
  echo "workers_failed"
  kill "$MONITOR_PID" 2>/dev/null || true
  exit "$status"
fi
"$PY" "$TOOL" --mode summarize --project-root "$P" --input-root "$IN" --output-root "$OUT" > "$OUT/telemetry/summarize.log" 2>&1
"$PY" "$TOOL" --mode merge --project-root "$P" --input-root "$IN" --output-root "$OUT" > "$OUT/telemetry/merge.log" 2>&1
kill "$MONITOR_PID" 2>/dev/null || true
echo "full_mask_transfer_done"
