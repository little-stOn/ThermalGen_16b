#!/usr/bin/env bash
# Copy immutable parent deltas into an independent artifact store.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 SOURCE_ARTIFACT_ROOT DESTINATION_ARTIFACT_ROOT" >&2
  exit 2
fi

source_root="$(realpath "$1")"
destination_root="$2"
if [[ ! -d "${source_root}" ]]; then
  echo "Missing source artifact root: ${source_root}" >&2
  exit 2
fi
if [[ -e "${destination_root}" ]]; then
  echo "Destination must not already exist: ${destination_root}" >&2
  exit 2
fi

readonly -a required=(
  "native16_ir_style_pilot1000_aligned_7gpu/checkpoints/final.pt"
  "native16_grounding_gentle_full_pilot300_v4/checkpoints/step_00000200.pt"
  "native16_rwtd_pilot200_v1/checkpoints/step_00000150.pt"
  "native16_radiometric_bridge_stage1_v1/checkpoint-008000.pt"
  "native16_style_adapter_affine_cont100_v5/final.pt"
)

for relative_path in "${required[@]}"; do
  if [[ ! -f "${source_root}/${relative_path}" ]]; then
    echo "Missing required parent artifact: ${source_root}/${relative_path}" >&2
    exit 2
  fi
done

for relative_path in "${required[@]}"; do
  install -d "${destination_root}/$(dirname "${relative_path}")"
  # --reflink=auto avoids duplicate blocks on compatible filesystems while
  # preserving an independently addressable destination path.
  cp --reflink=auto --preserve=mode,timestamps \
    "${source_root}/${relative_path}" "${destination_root}/${relative_path}"
done

cat <<EOF
Copied parent artifacts to: ${destination_root}
Set NATIVE16_PARENT_ARTIFACT_ROOT=${destination_root}
Then run:
  native16-verify-parents --cfg configs/recipes/native16_roi_cf.yaml
EOF
