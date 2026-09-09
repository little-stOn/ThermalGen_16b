#!/usr/bin/env python3
"""Pilot pseudo-bbox generation for the MS2 RGB/thermal dataset.

The pilot detects objects in left RGB frames with Grounding DINO, then transfers
RGB boxes into left thermal coordinates with the per-sequence MS2 calibration
and refined RGB depth. Raw data are never copied or modified.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps


DEFAULT_PROMPT = (
    "a person. a pedestrian. a car. a truck. a bus. a motorcycle. a bicycle."
)

CANONICAL_CLASSES = (
    "human.pedestrian",
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus",
    "vehicle.motorcycle",
    "vehicle.bicycle",
)


@dataclass(frozen=True)
class Sample:
    sequence: str
    frame_id: str
    rgb_path: str
    thermal_path: str
    depth_rgb_path: str | None
    depth_thr_path: str | None
    calib_path: str
    rgb_index: int
    thermal_index: int
    timestamp_rgb: float | None
    timestamp_thr: float | None
    timestamp_delta_ms: float | None
    review: bool


class GeometryFailure(RuntimeError):
    """A candidate cannot be transferred into a valid thermal box."""

    def __init__(self, reason: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics or {}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temp.replace(path)


def atomic_jsonl_dump(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            handle.write("\n")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def relative_to_project(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def resolve_project_path(value: str | None, project_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _last_number(line: str) -> float | None:
    tokens = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", line)
    if not tokens:
        return None
    try:
        return float(tokens[-1])
    except ValueError:
        return None


def read_timestamps(path: Path) -> list[float] | None:
    if not path.is_file():
        return None
    values: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            value = _last_number(stripped)
            if value is not None:
                values.append(value)
    return values or None


def timestamp_delta_ms(delta: float | None) -> float | None:
    if delta is None or not math.isfinite(delta):
        return None
    magnitude = abs(delta)
    # MS2 files may encode seconds, milliseconds, microseconds, or nanoseconds.
    if magnitude < 1.0:
        return delta * 1_000.0
    if magnitude < 1_000.0:
        return delta
    if magnitude < 1_000_000.0:
        return delta / 1_000.0
    return delta / 1_000_000.0


def numeric_sort_key(path: Path) -> tuple[int, float | str]:
    stem = path.stem
    try:
        return (0, float(stem))
    except ValueError:
        return (1, stem)


def image_files(directory: Path) -> list[Path]:
    return sorted((item for item in directory.glob("*.png") if item.is_file()), key=numeric_sort_key)


def find_depth_file(root: Path, sequence: str, sensor: str, frame_id: str) -> Path | None:
    candidates: list[Path] = []
    for base in (
        root / "proj_depth_refined" / sequence / sensor / "depth_refined",
        root / "proj_depth" / sequence / sensor,
    ):
        if not base.is_dir():
            continue
        candidates.extend(item for item in base.glob(f"{frame_id}.*") if item.is_file())
    priority = {".npy": 0, ".png": 1, ".tif": 2, ".tiff": 3}
    candidates.sort(key=lambda item: priority.get(item.suffix.lower(), 99))
    return candidates[0] if candidates else None


def nearest_timestamp_index(values: Sequence[float] | None, target: float | None, fallback: int) -> int:
    if not values or target is None:
        return min(fallback, max(0, len(values or [fallback]) - 1))
    # Timestamp streams are monotonic in MS2. Fall back to a full argmin if a
    # malformed stream is not monotonic.
    if any(values[index] > values[index + 1] for index in range(len(values) - 1)):
        return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - target)))
    insertion = bisect.bisect_left(values, target)
    candidates = [max(0, min(len(values) - 1, insertion + offset)) for offset in (-1, 0)]
    return min(candidates, key=lambda index: abs(values[index] - target))


def infer_project_root(dataset_root: Path) -> Path:
    # /.../<project>/data/raw/ms2 -> /.../<project>
    try:
        return dataset_root.resolve().parents[2]
    except IndexError as exc:
        raise ValueError("dataset root must be nested as <project>/data/raw/ms2") from exc


def build_manifest(
    dataset_root: Path,
    project_root: Path,
    frames_per_sequence: int,
    review_frames_per_sequence: int,
) -> list[Sample]:
    if frames_per_sequence < 1:
        raise ValueError("frames_per_sequence must be positive")
    if review_frames_per_sequence < 0:
        raise ValueError("review_frames_per_sequence cannot be negative")

    sync_root = dataset_root / "sync_data"
    if not sync_root.is_dir():
        raise FileNotFoundError(sync_root)
    sequences = sorted(item.name for item in sync_root.iterdir() if item.is_dir())
    if not sequences:
        raise RuntimeError(f"no MS2 sequences under {sync_root}")

    samples: list[Sample] = []
    for sequence in sequences:
        sequence_root = sync_root / sequence
        rgb_dir = sequence_root / "rgb" / "img_left"
        thr_dir = sequence_root / "thr" / "img_left"
        rgb_images = image_files(rgb_dir)
        thr_images = image_files(thr_dir)
        if not rgb_images or not thr_images:
            raise RuntimeError(f"missing RGB/thermal left images for {sequence}")

        rgb_timestamps = read_timestamps(sequence_root / "rgb" / "img_left_timestamp.txt")
        thr_timestamps = read_timestamps(sequence_root / "thr" / "img_left_timestamp.txt")
        positions = np.linspace(0, len(rgb_images) - 1, frames_per_sequence, dtype=np.int64)
        positions = sorted(set(int(position) for position in positions))
        review_positions = set(
            np.linspace(0, len(positions) - 1, review_frames_per_sequence, dtype=np.int64).tolist()
        ) if review_frames_per_sequence else set()
        calib_path = sequence_root / "calib.npy"
        if not calib_path.is_file():
            raise RuntimeError(f"missing calibration for {sequence}: {calib_path}")

        for sample_position, rgb_index in enumerate(positions):
            rgb_path = rgb_images[rgb_index]
            rgb_timestamp = (
                rgb_timestamps[rgb_index] if rgb_timestamps and rgb_index < len(rgb_timestamps) else None
            )
            thr_index = nearest_timestamp_index(thr_timestamps, rgb_timestamp, rgb_index)
            thr_path = thr_images[min(thr_index, len(thr_images) - 1)]
            thr_timestamp = (
                thr_timestamps[thr_index] if thr_timestamps and thr_index < len(thr_timestamps) else None
            )
            delta_ms = timestamp_delta_ms(
                rgb_timestamp - thr_timestamp
                if rgb_timestamp is not None and thr_timestamp is not None
                else None
            )
            frame_id = rgb_path.stem
            samples.append(
                Sample(
                    sequence=sequence,
                    frame_id=frame_id,
                    rgb_path=relative_to_project(rgb_path, project_root),
                    thermal_path=relative_to_project(thr_path, project_root),
                    depth_rgb_path=(
                        relative_to_project(depth_path, project_root)
                        if (depth_path := find_depth_file(dataset_root, sequence, "rgb", frame_id))
                        else None
                    ),
                    depth_thr_path=(
                        relative_to_project(depth_path, project_root)
                        if (depth_path := find_depth_file(dataset_root, sequence, "thr", thr_path.stem))
                        else None
                    ),
                    calib_path=relative_to_project(calib_path, project_root),
                    rgb_index=rgb_index,
                    thermal_index=thr_index,
                    timestamp_rgb=rgb_timestamp,
                    timestamp_thr=thr_timestamp,
                    timestamp_delta_ms=delta_ms,
                    review=sample_position in review_positions,
                )
            )
    return samples


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.ndim == 0:
        calibration = value.item()
    elif isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        calibration = value.reshape(-1)[0]
    else:
        calibration = value
    if not isinstance(calibration, dict):
        raise ValueError(f"unsupported calibration object in {path}: {type(calibration)!r}")

    def get_array(*keys: str) -> np.ndarray:
        for key in keys:
            if key in calibration:
                return np.asarray(calibration[key], dtype=np.float64).reshape(-1)
        raise KeyError(f"none of {keys!r} found in {path}")

    def get_matrix(*keys: str) -> np.ndarray:
        for key in keys:
            if key in calibration:
                return np.asarray(calibration[key], dtype=np.float64).reshape(3, 3)
        raise KeyError(f"none of {keys!r} found in {path}")

    k_rgb = get_matrix("K_rgbL", "K_rgb_left", "K_rgb")
    k_thr = get_matrix("K_thrL", "K_thr_left", "K_thr", "K_thermalL")
    r_nir_to_rgb = get_matrix("R_nir2rgb", "R_nir_to_rgb")
    r_nir_to_thr = get_matrix("R_nir2thr", "R_nir_to_thr", "R_nir2thermal")
    t_nir_to_rgb = get_array("T_nir2rgb", "T_nir_to_rgb")
    t_nir_to_thr = get_array("T_nir2thr", "T_nir_to_thr", "T_nir2thermal")
    r_rgb_to_thr = r_nir_to_thr @ r_nir_to_rgb.T
    t_rgb_to_thr = t_nir_to_thr - r_rgb_to_thr @ t_nir_to_rgb
    return k_rgb, k_thr, np.concatenate((r_rgb_to_thr, t_rgb_to_thr[:, None]), axis=1)


def load_depth(path: Path) -> tuple[np.ndarray, float]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        depth = np.asarray(np.load(path, mmap_mode="r"))
    elif suffix in {".png", ".tif", ".tiff"}:
        depth = np.asarray(Image.open(path))
    else:
        raise ValueError(f"unsupported depth format: {path}")
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape} from {path}")
    scale = 1.0 / 256.0 if "proj_depth_refined" not in path.parts else 1.0
    return depth.astype(np.float32, copy=False), scale


def _depth_grid(
    depth: np.ndarray,
    scale: float,
    bbox: Sequence[float],
    image_size: tuple[int, int],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_width, image_height = image_size
    depth_height, depth_width = depth.shape
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(image_width), x1))
    x2 = max(0.0, min(float(image_width), x2))
    y1 = max(0.0, min(float(image_height), y1))
    y2 = max(0.0, min(float(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        raise GeometryFailure("rgb_box_empty")
    dx1 = max(0, min(depth_width - 1, int(math.floor(x1 * depth_width / image_width))))
    dx2 = max(dx1 + 1, min(depth_width, int(math.ceil(x2 * depth_width / image_width))))
    dy1 = max(0, min(depth_height - 1, int(math.floor(y1 * depth_height / image_height))))
    dy2 = max(dy1 + 1, min(depth_height, int(math.ceil(y2 * depth_height / image_height))))
    area = max(1, (dx2 - dx1) * (dy2 - dy1))
    stride = max(1, int(math.ceil(math.sqrt(area / max_points))))
    ys, xs = np.mgrid[dy1:dy2:stride, dx1:dx2:stride]
    values = depth[ys, xs] * scale
    # Convert depth-grid pixel centers into the RGB calibration pixel frame.
    us = (xs.astype(np.float64) + 0.5) * image_width / depth_width - 0.5
    vs = (ys.astype(np.float64) + 0.5) * image_height / depth_height - 0.5
    valid = np.isfinite(values) & (values > 0.1) & (values < 250.0)
    return us[valid], vs[valid], values[valid].astype(np.float64, copy=False)


def transfer_rgb_box_to_thermal(
    rgb_bbox: Sequence[float],
    depth_rgb: np.ndarray,
    depth_scale: float,
    k_rgb: np.ndarray,
    k_thr: np.ndarray,
    rgb_to_thr: np.ndarray,
    rgb_size: tuple[int, int],
    thermal_size: tuple[int, int],
    max_depth_points: int = 4096,
) -> dict[str, Any]:
    us, vs, z_m = _depth_grid(depth_rgb, depth_scale, rgb_bbox, rgb_size, max_depth_points)
    diagnostics: dict[str, Any] = {
        "depth_valid_count": int(z_m.size),
        "depth_median_m": float(np.median(z_m)) if z_m.size else None,
    }
    if z_m.size < 24:
        raise GeometryFailure("insufficient_rgb_depth", diagnostics)

    fx_rgb, fy_rgb = k_rgb[0, 0], k_rgb[1, 1]
    cx_rgb, cy_rgb = k_rgb[0, 2], k_rgb[1, 2]
    points_rgb_m = np.column_stack(
        ((us - cx_rgb) * z_m / fx_rgb, (vs - cy_rgb) * z_m / fy_rgb, z_m)
    )
    points_rgb_mm = points_rgb_m * 1000.0
    r_rgb_to_thr = rgb_to_thr[:, :3]
    t_rgb_to_thr = rgb_to_thr[:, 3]
    points_thr_mm = points_rgb_mm @ r_rgb_to_thr.T + t_rgb_to_thr
    z_thr_m = points_thr_mm[:, 2] / 1000.0
    valid_thr = np.isfinite(z_thr_m) & (z_thr_m > 0.1)
    if int(valid_thr.sum()) < 24:
        diagnostics["positive_thermal_depth_count"] = int(valid_thr.sum())
        raise GeometryFailure("insufficient_positive_thermal_depth", diagnostics)
    points_thr_mm = points_thr_mm[valid_thr]
    z_thr_m = z_thr_m[valid_thr]
    uv_h = points_thr_mm @ k_thr.T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    thermal_width, thermal_height = thermal_size
    inside = (
        np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < thermal_width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < thermal_height)
    )
    diagnostics.update(
        {
            "projected_count": int(uv.shape[0]),
            "projected_in_frame_count": int(inside.sum()),
            "projected_in_frame_fraction": float(inside.mean()),
            "thermal_depth_median_m": float(np.median(z_thr_m)),
        }
    )
    if int(inside.sum()) < 16 or float(inside.mean()) < 0.45:
        raise GeometryFailure("projection_out_of_frame", diagnostics)

    in_frame_uv = uv[inside]
    if in_frame_uv.shape[0] >= 20:
        low = np.percentile(in_frame_uv, 1.0, axis=0)
        high = np.percentile(in_frame_uv, 99.0, axis=0)
    else:
        low = in_frame_uv.min(axis=0)
        high = in_frame_uv.max(axis=0)
    x1, y1 = np.maximum(low, 0.0)
    x2, y2 = np.minimum(high, (thermal_width - 1.0, thermal_height - 1.0))
    if x2 - x1 < 3.0 or y2 - y1 < 3.0:
        raise GeometryFailure("thermal_box_too_small", diagnostics)
    diagnostics.update(
        {
            "bbox_thr_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "depth_p05_m": float(np.percentile(z_thr_m, 5.0)),
            "depth_p95_m": float(np.percentile(z_thr_m, 95.0)),
            "thermal_box_width": float(x2 - x1),
            "thermal_box_height": float(y2 - y1),
        }
    )
    return diagnostics


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def canonical_class(label: str) -> str | None:
    value = label.lower().replace("_", " ")
    if "pedestrian" in value or "person" in value:
        return "human.pedestrian"
    if "motorcycle" in value or "motorbike" in value:
        return "vehicle.motorcycle"
    if "bicycle" in value or "bike" in value:
        return "vehicle.bicycle"
    if "truck" in value:
        return "vehicle.truck"
    if "bus" in value:
        return "vehicle.bus"
    if "car" in value or "automobile" in value:
        return "vehicle.car"
    return None


def classwise_nms(detections: list[dict[str, Any]], iou_threshold: float = 0.65) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    by_class: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
        class_name = str(detection["class_name"])
        if any(
            bbox_iou(detection["bbox_rgb_xyxy"], prior["bbox_rgb_xyxy"]) >= iou_threshold
            for prior in by_class[class_name]
        ):
            continue
        by_class[class_name].append(detection)
        kept.append(detection)
    return kept


def open_thermal_for_display(path: Path) -> Image.Image:
    image = Image.open(path)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.uint8:
        return Image.fromarray(array, mode="L").convert("RGB")
    finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.floating) else array.reshape(-1)
    if finite.size == 0:
        scaled = np.zeros(array.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite.astype(np.float32), [1.0, 99.0])
        if not math.isfinite(float(low)) or high <= low:
            low, high = float(finite.min()), float(finite.max())
        scaled = np.clip((array.astype(np.float32) - low) * 255.0 / max(high - low, 1e-6), 0, 255)
        scaled = scaled.astype(np.uint8)
    return ImageOps.autocontrast(Image.fromarray(scaled, mode="L")).convert("RGB")


def draw_box(draw: ImageDraw.ImageDraw, bbox: Sequence[float], color: tuple[int, int, int], label: str, width: int = 3) -> None:
    coords = tuple(float(value) for value in bbox)
    for offset in range(width):
        draw.rectangle(
            (coords[0] - offset, coords[1] - offset, coords[2] + offset, coords[3] + offset),
            outline=color,
        )
    draw.text((coords[0] + 3, max(0.0, coords[1] - 14.0)), label, fill=color)


def save_visualization(sample: dict[str, Any], detections: list[dict[str, Any]], output_path: Path, project_root: Path) -> None:
    rgb_path = resolve_project_path(sample["rgb_path"], project_root)
    thermal_path = resolve_project_path(sample["thermal_path"], project_root)
    if rgb_path is None or thermal_path is None:
        return
    rgb = Image.open(rgb_path).convert("RGB")
    thermal = open_thermal_for_display(thermal_path)
    scale = rgb.height / max(1, thermal.height)
    thermal = thermal.resize((round(thermal.width * scale), rgb.height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (rgb.width + thermal.width, rgb.height), "black")
    canvas.paste(rgb, (0, 0))
    canvas.paste(thermal, (rgb.width, 0))
    draw = ImageDraw.Draw(canvas)
    header = f"{sample['sequence']}  frame={sample['frame_id']}  detections={len(detections)}"
    draw.text((5, 5), header, fill=(255, 255, 0))
    for detection in detections:
        label = f"{detection['class_name']} {float(detection['score']):.2f}"
        draw_box(draw, detection["bbox_rgb_xyxy"], (0, 255, 0), label)
        tx1, ty1, tx2, ty2 = detection["bbox_thr_xyxy"]
        scale_x = thermal.width / max(1.0, float(sample["thermal_width"]))
        scale_y = thermal.height / max(1.0, float(sample["thermal_height"]))
        draw_box(
            draw,
            (rgb.width + tx1 * scale_x, ty1 * scale_y, rgb.width + tx2 * scale_x, ty2 * scale_y),
            (255, 80, 0),
            label,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=95, optimize=True)


def load_detector(model_id: str, cache_dir: Path, device: str):
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_dir / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))
    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("pilot requires torch and transformers in the labeling environment") from exc

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = model.to(device)
    model.eval()
    return processor, model, torch


def result_label(value: Any, prompts: Sequence[str], torch_module: Any) -> str:
    if isinstance(value, torch_module.Tensor):
        index = int(value.item())
        return prompts[index] if 0 <= index < len(prompts) else str(index)
    return str(value)


def run_inference(
    manifest: list[Sample],
    output_root: Path,
    project_root: Path,
    model_id: str,
    cache_dir: Path,
    device: str,
    box_threshold: float,
    text_threshold: float,
    prompt: str,
    nms_iou: float,
) -> None:
    processor, model, torch = load_detector(model_id, cache_dir, device)
    prompts = [part.strip() for part in prompt.split(".") if part.strip()]
    text = ". ".join(prompts) + "."
    manifest_rows = [asdict(sample) for sample in manifest]
    atomic_jsonl_dump(manifest_rows, output_root / "pilot_manifest.jsonl")
    atomic_json_dump(
        {
            "model_id": model_id,
            "prompt": text,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "nms_iou": nms_iou,
            "device": device,
            "frames": len(manifest),
            "sequences": sorted({sample.sequence for sample in manifest}),
            "coordinate_contract": "bbox_thr_xyxy is in left thermal image pixels; bbox_rgb_xyxy is in left RGB pixels",
            "geometry_contract": "RGB refined depth in meters; MS2 calibration translation in millimeters",
            "manual_review_required": True,
            "full_scale_started": False,
        },
        output_root / "metadata.json",
    )

    calibration_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    frame_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    visualization_root = output_root / "visualizations" / "review"

    for ordinal, sample in enumerate(manifest, start=1):
        sample_dict = asdict(sample)
        rgb_path = resolve_project_path(sample.rgb_path, project_root)
        thermal_path = resolve_project_path(sample.thermal_path, project_root)
        depth_path = resolve_project_path(sample.depth_rgb_path, project_root)
        depth_thr_path = resolve_project_path(sample.depth_thr_path, project_root)
        calib_path = resolve_project_path(sample.calib_path, project_root)
        if not rgb_path or not thermal_path or not calib_path:
            raise RuntimeError(f"incomplete manifest row: {sample}")
        rgb_image = Image.open(rgb_path).convert("RGB")
        thermal_image = Image.open(thermal_path)
        rgb_size = rgb_image.size
        thermal_size = thermal_image.size
        sample_dict.update({"rgb_width": rgb_size[0], "rgb_height": rgb_size[1], "thermal_width": thermal_size[0], "thermal_height": thermal_size[1]})
        if calib_path not in calibration_cache:
            calibration_cache[calib_path] = load_calibration(calib_path)
        k_rgb, k_thr, rgb_to_thr = calibration_cache[calib_path]

        inputs = processor(images=rgb_image, text=text, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        try:
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[rgb_image.size[::-1]],
            )[0]
        except TypeError:
            result = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[rgb_image.size[::-1]],
            )[0]

        labels = result.get("labels", result.get("text_labels", []))
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        candidates: list[dict[str, Any]] = []
        for box, score, label_value in zip(boxes, scores, labels):
            label = result_label(label_value, prompts, torch)
            class_name = canonical_class(label)
            if class_name is None:
                continue
            rgb_bbox = [float(value) for value in box.tolist()]
            candidates.append(
                {
                    "class_name": class_name,
                    "detector_label": label,
                    "score": float(score),
                    "bbox_rgb_xyxy": rgb_bbox,
                }
            )
        candidates = classwise_nms(candidates, nms_iou)

        depth_rgb: np.ndarray | None = None
        depth_scale = 1.0
        if depth_path is not None and depth_path.is_file():
            depth_rgb, depth_scale = load_depth(depth_path)
        else:
            rejection_rows.append({**sample_dict, "reason": "missing_rgb_depth"})
        depth_thr: np.ndarray | None = None
        thr_scale = 1.0
        if depth_thr_path is not None and depth_thr_path.is_file():
            depth_thr, thr_scale = load_depth(depth_thr_path)

        accepted: list[dict[str, Any]] = []
        for candidate in candidates:
            record = {**sample_dict, **candidate}
            try:
                if depth_rgb is None:
                    raise GeometryFailure("missing_rgb_depth")
                geometry = transfer_rgb_box_to_thermal(
                    candidate["bbox_rgb_xyxy"],
                    depth_rgb,
                    depth_scale,
                    k_rgb,
                    k_thr,
                    rgb_to_thr,
                    rgb_size,
                    thermal_size,
                )
                record.update(geometry)
                record["bbox_thr_xyxy"] = geometry.pop("bbox_thr_xyxy")
                record["source"] = "grounding_dino_rgb+ms2_calib+proj_depth_refined"
                if depth_thr is not None:
                    tx1, ty1, tx2, ty2 = record["bbox_thr_xyxy"]
                    x0 = max(0, int(math.floor(tx1)))
                    x1 = min(depth_thr.shape[1], int(math.ceil(tx2)))
                    y0 = max(0, int(math.floor(ty1)))
                    y1 = min(depth_thr.shape[0], int(math.ceil(ty2)))
                    if x1 > x0 and y1 > y0:
                        thermal_values = depth_thr[y0:y1, x0:x1] * thr_scale
                        thermal_valid = thermal_values[
                            np.isfinite(thermal_values) & (thermal_values > 0.1)
                        ]
                        record["thermal_box_depth_valid_count"] = int(thermal_valid.size)
                        record["thermal_box_depth_median_m"] = (
                            float(np.median(thermal_valid)) if thermal_valid.size else None
                        )
                accepted.append(record)
            except GeometryFailure as failure:
                rejection_rows.append({**record, "reason": failure.reason, **failure.diagnostics})
        detection_rows.extend(accepted)
        frame_rows.append(
            {
                **sample_dict,
                "status": "accepted" if accepted else ("detector_empty" if not candidates else "geometry_rejected"),
                "candidate_count": len(candidates),
                "accepted_count": len(accepted),
            }
        )
        if sample.review:
            filename = f"{ordinal:05d}_{sample.sequence}_{sample.frame_id}.jpg"
            save_visualization(sample_dict, accepted, visualization_root / filename, project_root)
        if ordinal % 25 == 0 or ordinal == len(manifest):
            print(f"processed {ordinal}/{len(manifest)} frames; accepted boxes={len(detection_rows)}", flush=True)

    atomic_jsonl_dump(frame_rows, output_root / "pilot_frames.jsonl")
    atomic_jsonl_dump(detection_rows, output_root / "pilot_detections.jsonl")
    atomic_jsonl_dump(rejection_rows, output_root / "pilot_rejections.jsonl")
    write_coco_dataset(output_root)
    write_quality_report(output_root)


def write_coco_dataset(output_root: Path) -> None:
    frame_rows = read_jsonl(output_root / "pilot_frames.jsonl")
    detection_rows = read_jsonl(output_root / "pilot_detections.jsonl")
    image_id_by_key: dict[tuple[str, str], int] = {}
    images: list[dict[str, Any]] = []
    for image_id, row in enumerate(frame_rows, start=1):
        key = (str(row["sequence"]), str(row["frame_id"]))
        image_id_by_key[key] = image_id
        images.append(
            {
                "id": image_id,
                "file_name": row["thermal_path"],
                "width": int(row["thermal_width"]),
                "height": int(row["thermal_height"]),
                "sequence": row["sequence"],
                "frame_id": row["frame_id"],
            }
        )
    category_ids = {name: index for index, name in enumerate(CANONICAL_CLASSES, start=1)}
    annotations: list[dict[str, Any]] = []
    for annotation_id, row in enumerate(detection_rows, start=1):
        image_id = image_id_by_key[(str(row["sequence"]), str(row["frame_id"]))]
        x1, y1, x2, y2 = (float(value) for value in row["bbox_thr_xyxy"])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[str(row["class_name"])],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "score": float(row["score"]),
                "source": row["source"],
                "bbox_rgb_xyxy": row["bbox_rgb_xyxy"],
                "depth_valid_count": row["depth_valid_count"],
            }
        )
    categories = [
        {
            "id": category_ids[name],
            "name": name,
            "supercategory": name.split(".", maxsplit=1)[0],
        }
        for name in CANONICAL_CLASSES
    ]
    atomic_json_dump(
        {"images": images, "annotations": annotations, "categories": categories},
        output_root / "pilot_coco.json",
    )


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    quantile_values = np.percentile(np.asarray(values, dtype=np.float64), [0, 25, 50, 75, 95, 100])
    return {
        "min": float(quantile_values[0]),
        "p25": float(quantile_values[1]),
        "median": float(quantile_values[2]),
        "p75": float(quantile_values[3]),
        "p95": float(quantile_values[4]),
        "max": float(quantile_values[5]),
    }


def write_quality_report(output_root: Path) -> None:
    frame_rows = read_jsonl(output_root / "pilot_frames.jsonl")
    detection_rows = read_jsonl(output_root / "pilot_detections.jsonl")
    rejection_rows = read_jsonl(output_root / "pilot_rejections.jsonl")
    class_counts = Counter(str(row["class_name"]) for row in detection_rows)
    sequence_counts = Counter(str(row["sequence"]) for row in detection_rows)
    reject_counts = Counter(str(row["reason"]) for row in rejection_rows)
    report = {
        "status": "pilot_complete",
        "frames_total": len(frame_rows),
        "frames_with_candidates": sum(row["candidate_count"] > 0 for row in frame_rows),
        "frames_with_accepted_boxes": sum(row["accepted_count"] > 0 for row in frame_rows),
        "detections_total": len(detection_rows),
        "accepted_by_class": dict(sorted(class_counts.items())),
        "accepted_by_sequence": dict(sorted(sequence_counts.items())),
        "score": _quantiles([float(row["score"]) for row in detection_rows]),
        "rgb_depth_support": _quantiles([float(row["depth_valid_count"]) for row in detection_rows]),
        "thermal_depth_support": _quantiles(
            [float(row["thermal_box_depth_valid_count"]) for row in detection_rows if "thermal_box_depth_valid_count" in row]
        ),
        "timestamp_delta_ms": _quantiles(
            [float(row["timestamp_delta_ms"]) for row in frame_rows if row.get("timestamp_delta_ms") is not None]
        ),
        "rejected_by_reason": dict(sorted(reject_counts.items())),
        "review_visualizations": len(list((output_root / "visualizations" / "review").glob("*.jpg"))),
        "manual_review_required": True,
        "automated_precision_claim": False,
        "full_scale_started": False,
        "next_gate": "manual review of review visualizations; tune thresholds and geometry before full scale",
    }
    atomic_json_dump(report, output_root / "quality_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def command_pilot(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else infer_project_root(dataset_root)
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        dataset_root,
        project_root,
        args.frames_per_sequence,
        args.review_frames_per_sequence,
    )
    cache_dir = Path(args.cache_dir).resolve()
    run_inference(
        manifest,
        output_root,
        project_root,
        args.model_id,
        cache_dir,
        args.device,
        args.box_threshold,
        args.text_threshold,
        args.prompt,
        args.nms_iou,
    )


def command_manifest(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else infer_project_root(dataset_root)
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output_root}")
    manifest = build_manifest(
        dataset_root,
        project_root,
        args.frames_per_sequence,
        args.review_frames_per_sequence,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_jsonl_dump([asdict(sample) for sample in manifest], output_root / "pilot_manifest.jsonl")
    print(f"wrote {len(manifest)} manifest rows to {output_root / 'pilot_manifest.jsonl'}")


def command_report(args: argparse.Namespace) -> None:
    write_quality_report(Path(args.output_root).resolve())


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset-root", required=True)
        subparser.add_argument("--project-root")
        subparser.add_argument("--output-root", required=True)
        subparser.add_argument("--frames-per-sequence", type=int, default=20)
        subparser.add_argument("--review-frames-per-sequence", type=int, default=5)
        subparser.add_argument("--overwrite", action="store_true")

    manifest_parser = subparsers.add_parser("manifest")
    add_common(manifest_parser)
    manifest_parser.set_defaults(func=command_manifest)

    pilot_parser = subparsers.add_parser("pilot")
    add_common(pilot_parser)
    pilot_parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    pilot_parser.add_argument("--cache-dir", required=True)
    pilot_parser.add_argument("--device", default="cuda:0")
    pilot_parser.add_argument("--box-threshold", type=float, default=0.35)
    pilot_parser.add_argument("--text-threshold", type=float, default=0.25)
    pilot_parser.add_argument("--nms-iou", type=float, default=0.65)
    pilot_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    pilot_parser.set_defaults(func=command_pilot)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--output-root", required=True)
    report_parser.set_defaults(func=command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
