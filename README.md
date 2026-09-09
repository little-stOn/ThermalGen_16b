# Native16 GLIGEN
[English](README.md) | [中文](README.zh-CN.md)

Reproducible training and inference for **box-conditioned, radiometrically meaningful 16-bit thermal diffusion**. This repository is an independent code extraction: it packages the Native16 branch, its configuration contract, tests, and command-line entry points without importing the parent experiment repository at runtime.

> The code is self-contained. Model weights, the GLIGEN base model, and the thermal dataset are deliberately external, versioned artifacts; they are not redistributed here.

## Scope

Included:

- native `uint16` TIFF loading, radiometric normalization, and lossless TIFF output;
- one-channel Native16 VAE adaptation and latent affine calibration;
- radiometric latent bridge, output calibration, and window-aware radiometric transport;
- GLIGEN box/text conditioning, scale-aware instance fusion, BoxDiff guidance, and model-stack loading;
- the ROI counterfactual training objective, distributed training, resumable delta checkpoints, and provenance checks;
- a pinned `native16_roi_cf` recipe, parent-stack manifest, smoke tests, and TIFF completeness validation.

Excluded intentionally: RGB8-only paths, detector training/inference, dataset builders, historical experiment scripts, datasets, base models, and checkpoint binaries.

## Method overview

```text
uint16 TIFF + radiometric profile
  -> normalized single-channel thermal tensor
  -> Native16 VAE / latent affine / radiometric bridge
  -> GLIGEN denoiser with text + normalized boxes
       + style UNet + grounding + RWT-D adapter model stack
       + scale-aware instance fusion
  -> radiometric output calibration + ROI-aware transport
  -> uint16 TIFF, preview, overlay, and metadata.json
```

Training uses diffusion noise prediction plus enabled auxiliary losses from `configs/recipes/native16_roi_cf.yaml`: instance-core, teacher distillation, and counterfactual grounding. Every saved delta records its parent model-stack fingerprint, so it cannot silently load on a different parent stack.

## Repository layout

```text
configs/
  recipes/native16_roi_cf.yaml          # reproducible training/inference recipe
  model_stacks/native16_roi_cf_init.yaml# immutable ordered parent deltas + SHA-256
  local/server.env.example              # host-specific path template
src/native16_gligen/
  train.py                              # distributed Native16 training
  sample_native16.py                    # Native16 uint16 TIFF generation
  model_stack.py                         # ordered delta loading + key contracts
  thermal16.py                           # radiometric tensor/TIFF codec
  native16_vae.py                        # single-channel VAE adaptation
  radiometric_*.py                       # bridge, calibration, transport
  instance_fusion.py                     # ROI/scale-aware GLIGEN fusion
scripts/
  train.sh                              # torchrun training entry point
  generate.sh                           # inference entry point
  copy_parent_artifacts.sh              # explicit one-time asset migration
CLI: native16-verify-parents            # hashes the pinned parent stack
examples/native16_sample_requests.json  # request schema example
tests/                                  # fast unit and configuration contracts
```

## Model-agnostic thermal video pipeline

`src/dwm/pipelines/` provides the complete training/inference lifecycle without
embedding the production model.  `ThermalVideoBatch.from_loader` adapts the
current loader fields (`vae_images`, `box_condition_images`, nested annotations,
`box_image_sizes`, and `pts`) to the canonical `[B,T,V,...]` contract.  Declared
second- or millisecond-based timestamps are normalized to seconds; legacy
batches without `pts_unit` are interpreted as seconds.  Pixel boxes remain
paired with their original `[height, width]`.

The future model integration implements `ThermalVideoModel` and receives the
whole `[B,T,V]` clip in `forward_video`.  The pipeline owns diffusion
training, valid-unit weighted accumulation, distributed reduction, classifier-
free guidance, preview/evaluation callbacks, and update-boundary checkpoints.
Codec details and TIFF/video writing stay behind the injected model and
`VideoWriter`; no concrete new-model implementation is included.

```python
from dwm.common import load_pipeline_from_config

pipeline, config = load_pipeline_from_config(
    "/path/to/pipeline.yaml",
    model=new_video_model,
    optimizer=optimizer,
)
```

The configuration's `pipeline._class_name` points to
`dwm.pipelines.ThermalVideoPipeline`; runtime dependencies are injected so the
model-specific interface remains separate from dataset orchestration.

## Installation

Use Python 3.10+ and install a CUDA-compatible PyTorch build first. The project intentionally does not pin `torch` because the correct wheel depends on the CUDA driver and platform.

```bash
conda create -n native16-gligen python=3.10 -y
conda activate native16-gligen
# Choose the wheel index appropriate for the target CUDA runtime.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[train,eval,test]'
cp configs/local/server.env.example configs/local/server.env
```

Set absolute paths in `configs/local/server.env`:

```bash
export NATIVE16_MODEL_ROOT=/srv/models
export NATIVE16_DATA_ROOT=/srv/datasets
export NATIVE16_ARTIFACT_ROOT=/srv/native16-gligen-artifacts
export NATIVE16_PARENT_ARTIFACT_ROOT=/srv/native16-parent-artifacts
export NATIVE16_PY=/path/to/venv/bin/python
export NATIVE16_TORCHRUN=/path/to/venv/bin/torchrun
export NATIVE16_TRAIN_GPUS=8
export NATIVE16_EVAL_GPUS="0 1"
```

`NATIVE16_ARTIFACT_ROOT` is writable output only. `NATIVE16_PARENT_ARTIFACT_ROOT` is a read-only store of the immutable parent deltas. Keeping these roots separate prevents training output from overwriting inference prerequisites.

## Required external assets

### Base model

`NATIVE16_MODEL_ROOT/gligen-1-4-generation-text-box` must be a complete Diffusers GLIGEN SD1.4 pipeline containing `vae/config.json`. The code rejects a path that is not a Diffusers base-model directory.

### Parent artifacts

The checked-in stack manifest pins three ordered UNet deltas; the recipe also requires a radiometric bridge and a style adapter:

| Role | Required relative path |
|---|---|
| Style UNet | `native16_ir_style_pilot1000_aligned_7gpu/checkpoints/final.pt` |
| Grounding | `native16_grounding_gentle_full_pilot300_v4/checkpoints/step_00000200.pt` |
| Grounding adapter | `native16_rwtd_pilot200_v1/checkpoints/step_00000150.pt` |
| Radiometric bridge | `native16_radiometric_bridge_stage1_v1/checkpoint-008000.pt` |
| Style adapter | `native16_style_adapter_affine_cont100_v5/final.pt` |

`configs/model_stacks/native16_roi_cf_init.yaml` contains SHA-256 digests and key contracts for the three stack layers. Do not substitute checkpoints without making a new audited manifest.

To make the deployment independent from the parent artifact directory, explicitly copy the five assets once:

```bash
scripts/copy_parent_artifacts.sh /path/to/legacy-artifacts /srv/native16-parent-artifacts
export NATIVE16_PARENT_ARTIFACT_ROOT=/srv/native16-parent-artifacts
native16-verify-parents --cfg configs/recipes/native16_roi_cf.yaml
```

The copy tool refuses an existing destination and uses copy-on-write reflinks when the filesystem supports them. It never mutates the source artifacts. The parent style checkpoint can be large; verify storage capacity before running it.

### Dataset contract

The supplied recipe targets the `FLIR_ADAS_v2` Native16 training manifest:

```text
${NATIVE16_DATA_ROOT}/infrared_uav/detector_ir16_v1/
${NATIVE16_DATA_ROOT}/infrared_uav/metadata_full_v1/flir_ir16_train.jsonl
${NATIVE16_DATA_ROOT}/infrared_uav/detector_ir16_v1/radiometric_profile.json
```

Each JSONL record must provide the `thermal16` field selected by the recipe, valid boxes, and labels from:

```text
person, bike, car, motor, bus, truck, other_vehicle
```

The dataset, its metadata, and usage rights remain the responsibility of the operator; no images or annotations are shipped in this repository.

## Reproducible training

Validate configuration and run a one-process dry run first:

```bash
source configs/local/server.env
export PYTHONPATH="$PWD/src"
"${NATIVE16_PY}" -m native16_gligen.train \
  --cfg configs/recipes/native16_roi_cf.yaml \
  --dry-run
```

Launch the configured distributed run:

```bash
scripts/train.sh configs/recipes/native16_roi_cf.yaml
```

For a bounded pilot without editing the recipe:

```bash
scripts/train.sh configs/recipes/native16_roi_cf.yaml \
  --set train.max_steps=300 \
  --set train.scheduler_total_steps=300
```

A training delta belongs to exactly one ordered parent stack. To resume, use the recipe's `train.resume` setting or a deliberate `--set train.resume=/absolute/path/to/checkpoint.pt`; do not replace stack layers through legacy checkpoint flags.

## Native16 inference

A request is a JSON list. Each item needs a unique `name`, a prompt, normalized `xyxy` boxes, and matching labels/phrases. See `examples/native16_sample_requests.json`.

```bash
# Inspect every resolved path and config field without loading a model.
scripts/generate.sh \
  --requests examples/native16_sample_requests.json \
  --output "$NATIVE16_ARTIFACT_ROOT/native16_samples/smoke" \
  --dry-run

# Generate Native16 TIFF files from the pinned parent stack.
scripts/generate.sh \
  --requests examples/native16_sample_requests.json \
  --output "$NATIVE16_ARTIFACT_ROOT/native16_samples/baseline" \
  --seed 20260812 \
  --guidance-scale 10.0 \
  --no-previews

# Evaluate a newly trained delta on the same audited parent stack.
scripts/generate.sh \
  --requests examples/native16_sample_requests.json \
  --checkpoint-layer /absolute/path/to/step_00000300.pt \
  --output "$NATIVE16_ARTIFACT_ROOT/native16_samples/step300"
```

Successful generation creates `images16/*.tiff`, optional `previews/*.png` and `overlays/*.png`, and `metadata.json`. The metadata captures the resolved parent-stack fingerprint, added checkpoint layer, VAE/bridge/calibration provenance, seed, boxes, phrases, and output ranges.

Verify that a generation directory contains exactly one TIFF per request:

```bash
native16-verify-tiffs \
  --requests examples/native16_sample_requests.json \
  --image-dir "$NATIVE16_ARTIFACT_ROOT/native16_samples/baseline/images16"
```

## Verification

Fast contracts do not require model weights or a dataset:

```bash
pytest -q
python -m compileall -q src tests
```

Before a production run, additionally run the training and inference dry runs above. Before reporting an inference result, retain `metadata.json` with the corresponding generated TIFF files.

## Reproducibility rules

1. Keep `configs/recipes/native16_roi_cf.yaml`, its resolved environment, the parent-stack manifest, and generated `resolved_config.yaml` together with each experiment.
2. Treat the parent-stack SHA-256 values and fingerprint as immutable. A changed layer requires a new stack manifest and a new experiment identity.
3. Do not convert source TIFF files through PNG/JPEG in the Native16 path. The model accepts radiometric single-channel data and writes `uint16` TIFF.
4. Use normalized `xyxy` request boxes for inference. Request names must remain unique after filename sanitization.
5. Keep base models, parent artifacts, input data, and writable outputs in separate directories.

## Upstream components

This extraction builds on the public GLIGEN and Diffusers ecosystems. Cite and comply with the licenses and model terms of every downloaded base model, dataset, and dependency in addition to this repository's MIT license.
