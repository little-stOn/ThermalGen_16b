#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${NATIVE16_ENV_FILE:-${ROOT}/configs/local/server.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

: "${NATIVE16_MODEL_ROOT:?Set NATIVE16_MODEL_ROOT in configs/local/server.env or the environment}"
: "${NATIVE16_DATA_ROOT:?Set NATIVE16_DATA_ROOT in configs/local/server.env or the environment}"
: "${NATIVE16_PARENT_ARTIFACT_ROOT:?Set NATIVE16_PARENT_ARTIFACT_ROOT to the immutable parent artifact store}"
: "${NATIVE16_ARTIFACT_ROOT:=${ROOT}/artifacts}"

PY="${NATIVE16_PY:-python}"
TORCHRUN="${NATIVE16_TORCHRUN:-torchrun}"
NPROC="${NATIVE16_TRAIN_GPUS:-1}"
CFG="${ROOT}/configs/recipes/native16_roi_cf.yaml"
if [[ $# -gt 0 ]]; then
  CFG="$1"
  shift
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${NATIVE16_ARTIFACT_ROOT}/logs"
"${TORCHRUN}" --standalone --nproc_per_node="${NPROC}" \
  -m native16_gligen.train --cfg "${CFG}" "$@" \
  2>&1 | tee "${NATIVE16_ARTIFACT_ROOT}/logs/$(basename "${CFG}" .yaml).log"
