#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${NATIVE16_ENV_FILE:-${ROOT}/configs/local/server.env}"
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

: "${NATIVE16_DATA_ROOT:?Set NATIVE16_DATA_ROOT in configs/local/server.env or the environment}"
: "${NATIVE16_MODEL_ROOT:?Set NATIVE16_MODEL_ROOT in configs/local/server.env or the environment}"
: "${NATIVE16_PARENT_ARTIFACT_ROOT:?Set NATIVE16_PARENT_ARTIFACT_ROOT to the immutable parent artifact store}"
: "${NATIVE16_ARTIFACT_ROOT:=${ROOT}/artifacts}"

PY="${NATIVE16_PY:-python}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PY}" -m native16_gligen.sample_native16 \
  --cfg "${ROOT}/configs/recipes/native16_roi_cf.yaml" "$@"
