#!/usr/bin/env python3
"""Pilot/full validation of RGB instance-mask to thermal bbox transfer.

This experiment is intentionally non-destructive. It reruns the existing SAM3.1
RGB detector with masks enabled, projects matched mask/depth pixels through the
MS2 calibration, optionally applies a bounded thermal-edge refinement, and
writes candidate metrics/previews. Production annotations are never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


CATEGORY_NAMES = (
    "human.pedestrian",
    "vehicle.bicycle",
    "vehicle.bus",
    "vehicle.car",
    "vehicle.motorcycle",
    "vehicle.truck",
)
PROMPTS = ("person", "car", "truck", "bus", "motorcycle", "bicycle")
PROMPT_TO_CLASS = {
    "person": "human.pedestrian",
    "car": "vehicle.car",
    "truck": "vehicle.truck",
    "bus": "vehicle.bus",
    "motorcycle": "vehicle.motorcycle",
    "bicycle": "vehicle.bicycle",
}
COLORS = {
    "human.pedestrian": (0, 255, 0),
    "vehicle.bicycle": (0, 220, 255),
    "vehicle.motorcycle": (0, 220, 255),
    "vehicle.car": (255, 80, 0),
    "vehicle.bus": (255, 220, 0),
    "vehicle.truck": (255, 220, 0),
}
METHODS = (
    "depth_cloud",
    "robust_depth_corners",
    "mask_depth_all",
    "mask_depth_foreground",
    "mask_depth_thermal_refine",
)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def frame_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["sequence"]), str(row["frame_id"])


def resolve_project_path(value: str | None, project_root: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def canonical_prompt(value: str) -> str | None:
    normalized = value.strip().lower()
    for prompt, class_name in PROMPT_TO_CLASS.items():
        if prompt in normalized:
            return class_name
    return None


def valid_box(box: Sequence[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def select_records(input_root: Path, frames_per_sequence: int) -> list[dict[str, Any]]:
    by_sequence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(input_root / "records.jsonl"):
        if row.get("boxes"):
            by_sequence[str(row["sequence"])].append(row)
    selected: list[dict[str, Any]] = []
    for sequence in sorted(by_sequence):
        rows = by_sequence[sequence]
        if frames_per_sequence <= 0 or frames_per_sequence >= len(rows):
            selected.extend(rows)
            continue
        positions = np.linspace(0, len(rows) - 1, frames_per_sequence, dtype=np.int64)
        selected.extend(rows[int(position)] for position in sorted(set(positions)))
    return selected


def import_project_helpers(project_root: Path):
    prepare_root = project_root / "scripts" / "prepare"
    if str(prepare_root) not in sys.path:
        sys.path.insert(0, str(prepare_root))
    from compare_ms2_bbox_transfer import (
        box_metrics,
        candidate_boxes,
        load_calibration,
        rgb_depth_quantile,
        thermal_display,
    )
    from ms2_pseudo_bbox import load_depth

    return box_metrics, candidate_boxes, load_calibration, rgb_depth_quantile, thermal_display, load_depth


def load_detector(args: argparse.Namespace):
    import torch

    sam3_eval_root = Path(args.sam3_eval_root).resolve()
    if str(sam3_eval_root) not in sys.path:
        sys.path.insert(0, str(sam3_eval_root))
    from sam31_detector import Sam31Detector

    config = {
        "input_resolution": int(args.input_resolution),
        "mask_resolution": int(args.mask_resolution),
        "save_masks": True,
        "prompts": list(PROMPTS),
        "confidence_threshold": float(args.model_confidence),
        "max_detections_per_prompt": 100,
        "max_detections_per_image": 150,
        "nms_iou_threshold": float(args.nms_iou),
        "mask_threshold": float(args.mask_threshold),
        "precision": str(args.precision),
        "checkpoint_minimum_coverage": 0.95,
        "checkpoint_mmap": True,
    }
    device = torch.device(args.device)
    detector = Sam31Detector(
        config=config,
        device=device,
        sam3_repo=str(sam3_eval_root / "sam3"),
        checkpoint=str(Path(args.checkpoint).resolve()),
    )
    return detector, torch, config


def load_uint16_model_input(path: Path, torch_module: Any):
    with Image.open(path) as source:
        array = np.asarray(source).astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"thermal image has no finite values: {path}")
    low, high = np.percentile(finite, [1.0, 99.0])
    scale = max(float(high - low), 1.0)
    normalized = np.clip((array - low) * 255.0 / scale, 0.0, 255.0).astype(np.uint8)
    rgb = np.repeat(normalized[..., None], 3, axis=2)
    return torch_module.from_numpy(rgb).permute(2, 0, 1).contiguous()


def match_predictions(
    source_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[int, dict[str, Any] | None]:
    by_class: defaultdict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, prediction in enumerate(predictions):
        class_name = canonical_prompt(str(prediction.get("prompt", "")))
        if class_name is not None:
            by_class[class_name].append((index, prediction))
    assignments: dict[int, dict[str, Any] | None] = {}
    used: set[int] = set()
    for source_index, source in sorted(
        enumerate(source_rows), key=lambda pair: float(pair[1]["score"]), reverse=True
    ):
        class_name = str(source["class_name"])
        choices = []
        for prediction_index, prediction in by_class.get(class_name, []):
            if prediction_index in used:
                continue
            overlap = bbox_iou(source["bbox_rgb_xyxy"], prediction["bbox_xyxy"])
            score_delta = abs(float(source["score"]) - float(prediction["score"]))
            choices.append((overlap, -score_delta, prediction_index, prediction))
        if not choices:
            assignments[source_index] = None
            continue
        overlap, _, prediction_index, prediction = max(choices, key=lambda value: (value[0], value[1]))
        if overlap < 0.30:
            assignments[source_index] = None
            continue
        used.add(prediction_index)
        assignments[source_index] = prediction
    return assignments


def project_mask_depth(
    mask: np.ndarray,
    rgb_bbox: Sequence[float],
    depth: np.ndarray,
    depth_scale: float,
    k_rgb: np.ndarray,
    k_thr: np.ndarray,
    rgb_to_thr: np.ndarray,
    rgb_size: tuple[int, int],
    thermal_size: tuple[int, int],
    mode: str,
) -> tuple[list[float] | None, dict[str, Any]]:
    rgb_width, rgb_height = rgb_size
    thermal_width, thermal_height = thermal_size
    mask_height, mask_width = mask.shape
    rows, columns = np.nonzero(mask)
    if rows.size > 4096:
        sample_indices = np.linspace(0, rows.size - 1, 4096, dtype=np.int64)
        rows, columns = rows[sample_indices], columns[sample_indices]
    diagnostics: dict[str, Any] = {
        "mask_pixel_count": int(rows.size),
        "mask_depth_valid_count": 0,
        "mask_projected_count": 0,
        "mask_projected_in_frame_count": 0,
        "mask_projected_in_frame_fraction": 0.0,
    }
    if rows.size < 16:
        return None, diagnostics
    # Restrict the segmentation mask to the original RGB detection box.
    mask_x = (columns.astype(np.float64) + 0.5) * rgb_width / mask_width - 0.5
    mask_y = (rows.astype(np.float64) + 0.5) * rgb_height / mask_height - 0.5
    x1, y1, x2, y2 = (float(value) for value in rgb_bbox)
    inside_box = (mask_x >= x1) & (mask_x <= x2) & (mask_y >= y1) & (mask_y <= y2)
    mask_x, mask_y = mask_x[inside_box], mask_y[inside_box]
    if mask_x.size < 16:
        return None, diagnostics
    depth_height, depth_width = depth.shape
    depth_x = np.clip(np.floor((mask_x + 0.5) * depth_width / rgb_width).astype(np.int64), 0, depth_width - 1)
    depth_y = np.clip(np.floor((mask_y + 0.5) * depth_height / rgb_height).astype(np.int64), 0, depth_height - 1)
    values = depth[depth_y, depth_x].astype(np.float64, copy=False) * float(depth_scale)
    finite = np.isfinite(values) & (values > 0.1) & (values < 200.0)
    diagnostics["mask_depth_valid_count"] = int(finite.sum())
    if int(finite.sum()) < 16:
        return None, diagnostics
    mask_x, mask_y, values = mask_x[finite], mask_y[finite], values[finite]
    if mode == "foreground":
        q50 = float(np.percentile(values, 50.0))
        keep = values <= q50
        if int(keep.sum()) < 16:
            keep = np.ones(values.shape, dtype=bool)
        mask_x, mask_y, values = mask_x[keep], mask_y[keep], values[keep]
    homogeneous = np.column_stack((mask_x, mask_y, np.ones(mask_x.size, dtype=np.float64)))
    rays = homogeneous @ np.linalg.inv(k_rgb).T
    points_rgb = rays * values[:, None]
    points_thr = points_rgb @ rgb_to_thr[:, :3].T + rgb_to_thr[:, 3] / 1000.0
    positive = np.isfinite(points_thr).all(axis=1) & (points_thr[:, 2] > 0.1)
    if int(positive.sum()) < 16:
        return None, diagnostics
    points_thr = points_thr[positive]
    uv_h = points_thr @ k_thr.T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    diagnostics["mask_projected_count"] = int(uv.shape[0])
    in_frame = (
        np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= 0.0) & (uv[:, 0] < thermal_width)
        & (uv[:, 1] >= 0.0) & (uv[:, 1] < thermal_height)
    )
    diagnostics["mask_projected_in_frame_count"] = int(in_frame.sum())
    diagnostics["mask_projected_in_frame_fraction"] = float(in_frame.mean()) if in_frame.size else 0.0
    if int(in_frame.sum()) < 16:
        return None, diagnostics
    uv = uv[in_frame]
    low = np.percentile(uv, 1.0, axis=0)
    high = np.percentile(uv, 99.0, axis=0)
    return valid_box([low[0], low[1], high[0], high[1]], thermal_width, thermal_height), diagnostics


def fast_thermal_metrics(
    box: Sequence[float],
    thermal: np.ndarray,
    gradient: np.ndarray,
    gradient_norm: float,
    robust_range: float,
) -> dict[str, float | int | None]:
    height, width = thermal.shape
    x1, y1, x2, y2 = (float(value) for value in box)
    left, right = int(round(x1)), int(round(x2))
    top, bottom = int(round(y1)), int(round(y2))
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    band = 2
    border_parts = [
        gradient[max(0, top - band) : min(height, top + band + 1), left:right],
        gradient[max(0, bottom - band - 1) : min(height, bottom + band), left:right],
        gradient[top:bottom, max(0, left - band) : min(width, left + band + 1)],
        gradient[top:bottom, max(0, right - band - 1) : min(width, right + band)],
    ]
    border_values = np.concatenate([part.reshape(-1) for part in border_parts if part.size])
    edge_alignment = float(np.mean(border_values) / gradient_norm) if border_values.size else 0.0
    margin_x = max(4, int(round(0.15 * (right - left))))
    margin_y = max(4, int(round(0.15 * (bottom - top))))
    outer_left, outer_right = max(0, left - margin_x), min(width, right + margin_x)
    outer_top, outer_bottom = max(0, top - margin_y), min(height, bottom + margin_y)
    inside = thermal[top:bottom, left:right].astype(np.float32)
    outer = thermal[outer_top:outer_bottom, outer_left:outer_right].astype(np.float32)
    ring_mask = np.ones(outer.shape, dtype=bool)
    ring_mask[top - outer_top : bottom - outer_top, left - outer_left : right - outer_left] = False
    ring = outer[ring_mask]
    contrast = (
        abs(float(np.median(inside)) - float(np.median(ring))) / robust_range
        if inside.size and ring.size
        else 0.0
    )
    return {
        "edge_alignment": edge_alignment,
        "thermal_contrast": contrast,
        "depth_support": None,
        "depth_purity": None,
    }


def refine_thermal_box(
    box: Sequence[float],
    thermal: np.ndarray,
    gradient: np.ndarray,
    gradient_norm: float,
    robust_range: float,
) -> tuple[list[float], tuple[int, int]]:
    x1, y1, x2, y2 = (float(value) for value in box)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    width = x2 - x1
    height = y2 - y1
    best = list(box)
    base = fast_thermal_metrics(best, thermal, gradient, gradient_norm, robust_range)
    best_score = float(base["edge_alignment"]) + 0.20 * float(base["thermal_contrast"])
    best_shift = (0, 0)
    for dx in range(-6, 7, 2):
        for dy in range(-6, 7, 2):
            for scale_x, scale_y in ((0.95, 0.95), (1.0, 1.0), (1.05, 1.05)):
                candidate = valid_box(
                    [
                        center_x + dx - width * scale_x / 2.0,
                        center_y + dy - height * scale_y / 2.0,
                        center_x + dx + width * scale_x / 2.0,
                        center_y + dy + height * scale_y / 2.0,
                    ],
                    thermal.shape[1],
                    thermal.shape[0],
                )
                if candidate is None:
                    continue
                metrics = fast_thermal_metrics(
                    candidate, thermal, gradient, gradient_norm, robust_range
                )
                size_penalty = 0.05 * (abs(math.log(scale_x)) + abs(math.log(scale_y)))
                shift_penalty = 0.015 * math.hypot(dx, dy)
                score = (
                    float(metrics["edge_alignment"])
                    + 0.20 * float(metrics["thermal_contrast"])
                    - size_penalty
                    - shift_penalty
                )
                if score > best_score:
                    best_score = score
                    best = candidate
                    best_shift = (dx, dy)
    return best, best_shift


def draw_panel(base: Image.Image, detections: Sequence[dict[str, Any]], boxes: Sequence[Sequence[float] | None], title: str) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), title, fill=(255, 255, 255))
    for detection, box in zip(detections, boxes):
        if box is None:
            continue
        draw.rectangle(tuple(box), outline=COLORS.get(str(detection["class_name"]), (255, 255, 255)), width=2)
    return image


def render_preview(
    output_path: Path,
    rgb_path: Path,
    thermal_image: Image.Image,
    detections: Sequence[dict[str, Any]],
    candidates: dict[str, Sequence[Sequence[float] | None]],
) -> None:
    with Image.open(rgb_path) as source:
        rgb = source.convert("RGB")
    rgb.thumbnail((640, 238), Image.Resampling.LANCZOS)
    rgb_panel = Image.new("RGB", (640, 256), "black")
    rgb_panel.paste(rgb, (0, 18))
    draw = ImageDraw.Draw(rgb_panel)
    draw.text((4, 3), "RGB detections", fill=(255, 255, 255))
    rgb_scale_x = rgb.width / 1224.0
    rgb_scale_y = rgb.height / 384.0
    for detection in detections:
        box = detection["bbox_rgb_xyxy"]
        draw.rectangle(
            (box[0] * rgb_scale_x, box[1] * rgb_scale_y + 18, box[2] * rgb_scale_x, box[3] * rgb_scale_y + 18),
            outline=COLORS.get(str(detection["class_name"]), (255, 255, 255)),
            width=2,
        )
    panels = [
        rgb_panel,
        thermal_image,
        draw_panel(thermal_image, detections, candidates["robust_depth_corners"], "D robust-depth-corners"),
        draw_panel(thermal_image, detections, candidates["mask_depth_all"], "M mask-depth all"),
        draw_panel(thermal_image, detections, candidates["mask_depth_foreground"], "N mask-depth foreground"),
        draw_panel(thermal_image, detections, candidates["mask_depth_thermal_refine"], "T mask-depth + thermal refine"),
    ]
    canvas = Image.new("RGB", (640 * 3, 256 * 2), "black")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 3) * 640, (index // 3) * 256))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=94)


def worker(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    shard_root = output_root / "shards" / ("rank%02d" % args.rank)
    shard_root.mkdir(parents=True, exist_ok=True)
    box_metrics, candidate_boxes, load_calibration, rgb_depth_quantile, thermal_display, load_depth = import_project_helpers(project_root)

    selected = select_records(input_root, args.frames_per_sequence)
    selected = selected[args.rank :: args.world_size]
    selected_keys = {frame_key(row) for row in selected}
    source_by_key: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source_index, row in enumerate(read_jsonl(input_root / "annotations.jsonl")):
        if frame_key(row) in selected_keys:
            source_by_key[frame_key(row)].append({**row, "source_line_index": source_index})

    detector, torch_module, detector_config = load_detector(args)
    selected_annotation_count = sum(len(source_by_key[frame_key(record)]) for record in selected)
    if selected_annotation_count == 0:
        raise RuntimeError("selected frames have no source annotations")

    shard_annotations_path = shard_root / "annotations.jsonl"
    shard_metrics_path = shard_root / "metrics.jsonl"
    annotations_handle = shard_annotations_path.open("w", encoding="utf-8")
    metrics_handle = shard_metrics_path.open("w", encoding="utf-8")
    annotation_count = 0
    metric_count = 0
    mask_match_count = 0
    preview_records: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[list[float] | None]], Image.Image, Path]] = []
    batch_size = max(1, int(args.batch_size))
    active_prompts: tuple[str, ...] | None = None
    for batch_start in range(0, len(selected), batch_size):
        batch_records = selected[batch_start : batch_start + batch_size]
        # The detector is deliberately run on RGB images, not thermal images.
        rgb_images = []
        for record in batch_records:
            rgb_path = resolve_project_path(record["rgb_path"], project_root)
            if rgb_path is None or not rgb_path.is_file():
                raise FileNotFoundError(record["rgb_path"])
            with Image.open(rgb_path) as source:
                rgb_array = np.asarray(source.convert("RGB")).copy()
            rgb_images.append(torch_module.from_numpy(rgb_array).permute(2, 0, 1).contiguous())
        batch_prompts = tuple(
            prompt
            for prompt in PROMPTS
            if any(
                PROMPT_TO_CLASS[prompt] == str(row["class_name"])
                for record in batch_records
                for row in source_by_key[frame_key(record)]
            )
        )
        if not batch_prompts:
            raise RuntimeError("selected batch has no supported source prompts")
        if batch_prompts != active_prompts:
            detector.prompts = list(batch_prompts)
            with torch_module.inference_mode():
                detector.text_outputs = detector.model.backbone.forward_text(
                    list(batch_prompts), device=detector.device
                )
            active_prompts = batch_prompts
        predictions_batch = detector.predict_batch(
            rgb_images,
            [{"height": int(source_by_key[frame_key(record)][0]["rgb_height"]), "width": int(source_by_key[frame_key(record)][0]["rgb_width"])} for record in batch_records],
        )

        for record, predictions in zip(batch_records, predictions_batch, strict=True):
            key = frame_key(record)
            detections = sorted(source_by_key[key], key=lambda row: float(row["score"]), reverse=True)
            if args.max_boxes_per_frame > 0:
                detections = detections[: args.max_boxes_per_frame]
            assignments = match_predictions(detections, predictions)
            depth_path = resolve_project_path(detections[0]["depth_rgb_path"], project_root)
            calib_path = resolve_project_path(detections[0]["calib_path"], project_root)
            thermal_path = resolve_project_path(record["image_path"], project_root)
            if depth_path is None or calib_path is None or thermal_path is None:
                raise FileNotFoundError("incomplete frame resources: %s" % (key,))
            depth, depth_scale = load_depth(depth_path)
            depth = depth * depth_scale
            k_rgb, k_thr, rgb_to_thr = load_calibration(calib_path)
            with Image.open(thermal_path) as thermal_source:
                thermal_array = np.asarray(thermal_source).astype(np.float32)
            thermal_image, gradient, gradient_norm = thermal_display(thermal_array)
            finite_thermal = thermal_array[np.isfinite(thermal_array)]
            thermal_low, thermal_high = np.percentile(finite_thermal, [1.0, 99.0])
            thermal_robust_range = max(float(thermal_high - thermal_low), 1e-6)
            candidates: dict[str, list[list[float] | None]] = {method: [] for method in METHODS}
            transformed_for_frame: list[dict[str, Any]] = []
            for source_index, detection in enumerate(detections):
                prediction = assignments.get(source_index)
                mask = None if prediction is None else prediction.get("mask")
                if args.final_only and mask is not None:
                    base_candidates = {}
                    d_box = None
                else:
                    base_candidates, _, _ = candidate_boxes(
                        detection,
                        depth,
                        k_rgb,
                        k_thr,
                        rgb_to_thr,
                        (int(detection["rgb_width"]), int(detection["rgb_height"])),
                        (int(detection["thermal_width"]), int(detection["thermal_height"])),
                    )
                    d_box = base_candidates.get(
                        "robust_depth_corners", base_candidates.get("depth_cloud")
                    )
                if mask is not None:
                    mask_array = np.asarray(mask, dtype=bool)
                    if args.final_only:
                        all_box = None
                    else:
                        all_box, all_diag = project_mask_depth(
                            mask_array,
                            detection["bbox_rgb_xyxy"],
                            depth,
                            1.0,
                            k_rgb,
                            k_thr,
                            rgb_to_thr,
                            (int(detection["rgb_width"]), int(detection["rgb_height"])),
                            (int(detection["thermal_width"]), int(detection["thermal_height"])),
                            "all",
                        )
                    foreground_box, foreground_diag = project_mask_depth(
                        mask_array,
                        detection["bbox_rgb_xyxy"],
                        depth,
                        1.0,
                        k_rgb,
                        k_thr,
                        rgb_to_thr,
                        (int(detection["rgb_width"]), int(detection["rgb_height"])),
                        (int(detection["thermal_width"]), int(detection["thermal_height"])),
                        "foreground",
                    )
                    if args.final_only:
                        all_diag = foreground_diag
                else:
                    all_box, foreground_box = None, None
                    all_diag = foreground_diag = {"mask_pixel_count": 0, "mask_depth_valid_count": 0, "mask_projected_count": 0, "mask_projected_in_frame_count": 0, "mask_projected_in_frame_fraction": 0.0}
                if foreground_box is None and args.final_only:
                    base_candidates, _, _ = candidate_boxes(
                        detection,
                        depth,
                        k_rgb,
                        k_thr,
                        rgb_to_thr,
                        (int(detection["rgb_width"]), int(detection["rgb_height"])),
                        (int(detection["thermal_width"]), int(detection["thermal_height"])),
                    )
                    d_box = base_candidates.get(
                        "robust_depth_corners", base_candidates.get("depth_cloud")
                    )
                    all_box = base_candidates.get("depth_cloud")
                if all_box is None:
                    all_box = d_box
                if foreground_box is None:
                    foreground_box = all_box
                if foreground_box is None:
                    rgb_box = detection["bbox_rgb_xyxy"]
                    scaled_box = [
                        float(rgb_box[0]) * float(detection["thermal_width"]) / float(detection["rgb_width"]),
                        float(rgb_box[1]) * float(detection["thermal_height"]) / float(detection["rgb_height"]),
                        float(rgb_box[2]) * float(detection["thermal_width"]) / float(detection["rgb_width"]),
                        float(rgb_box[3]) * float(detection["thermal_height"]) / float(detection["rgb_height"]),
                    ]
                    foreground_box = valid_box(
                        scaled_box,
                        int(detection["thermal_width"]),
                        int(detection["thermal_height"]),
                    )
                if all_box is None:
                    all_box = foreground_box
                refined_box, shift = refine_thermal_box(
                    foreground_box,
                    thermal_array,
                    gradient,
                    gradient_norm,
                    thermal_robust_range,
                )
                candidates["depth_cloud"].append(base_candidates.get("depth_cloud"))
                candidates["robust_depth_corners"].append(d_box)
                candidates["mask_depth_all"].append(all_box)
                candidates["mask_depth_foreground"].append(foreground_box)
                candidates["mask_depth_thermal_refine"].append(refined_box)
                transformed = dict(detection)
                transformed["bbox_thr_xyxy_calibrated"] = list(detection["bbox_thr_xyxy"])
                transformed["bbox_thr_xyxy"] = refined_box
                transformed["bbox_thr_xyxy_robust_depth_corners"] = d_box
                transformed["bbox_thr_xyxy_mask_depth_all"] = all_box
                transformed["bbox_thr_xyxy_mask_depth_foreground"] = foreground_box
                transformed["bbox_thr_xyxy_mask_depth_thermal_refine"] = refined_box
                transformed["transfer_method"] = "mask_depth_thermal_refine"
                transformed["source"] = "sam3.1_rgb_mask+ms2_calib+proj_depth_refined+thermal_bounded_refine"
                transformed["mask_match_iou"] = (
                    bbox_iou(detection["bbox_rgb_xyxy"], prediction["bbox_xyxy"])
                    if prediction is not None else None
                )
                transformed["mask_prediction_score"] = (
                    float(prediction["score"]) if prediction is not None else None
                )
                transformed["mask_pixel_count"] = all_diag["mask_pixel_count"]
                transformed["mask_depth_valid_count"] = all_diag["mask_depth_valid_count"]
                transformed["mask_projected_count"] = all_diag["mask_projected_count"]
                transformed["mask_projected_in_frame_count"] = all_diag["mask_projected_in_frame_count"]
                transformed["mask_projected_in_frame_fraction"] = all_diag["mask_projected_in_frame_fraction"]
                transformed["thermal_refine_dx"] = int(shift[0])
                transformed["thermal_refine_dy"] = int(shift[1])
                transformed.pop("source_line_index", None)
                for field in (
                    "projected_count",
                    "projected_in_frame_count",
                    "projected_in_frame_fraction",
                    "thermal_depth_median_m",
                    "depth_p05_m",
                    "depth_p95_m",
                    "thermal_box_depth_valid_count",
                    "thermal_box_depth_median_m",
                ):
                    transformed.pop(field, None)
                annotation_row = {**transformed, "source_line_index": detection["source_line_index"]}
                annotations_handle.write(
                    json.dumps(annotation_row, ensure_ascii=False, separators=(",", ":"))
                )
                annotations_handle.write("\n")
                annotation_count += 1
                if prediction is not None:
                    mask_match_count += 1
                for method in METHODS:
                    box = candidates[method][-1]
                    if box is None:
                        continue
                    metrics = fast_thermal_metrics(
                        box,
                        thermal_array,
                        gradient,
                        gradient_norm,
                        thermal_robust_range,
                    )
                    metric_row = {
                        "sequence": record["sequence"],
                        "frame_id": record["frame_id"],
                        "class_name": detection["class_name"],
                        "source_line_index": detection["source_line_index"],
                        "method": method,
                        "bbox_xyxy": box,
                        **metrics,
                    }
                    metrics_handle.write(
                        json.dumps(metric_row, ensure_ascii=False, separators=(",", ":"))
                    )
                    metrics_handle.write("\n")
                    metric_count += 1
                transformed_for_frame.append(detection)
            if len(preview_records) < int(args.preview_count) and (batch_start + len(preview_records)) % max(1, int(args.preview_stride)) == 0:
                preview_records.append((record, transformed_for_frame, candidates, thermal_image, rgb_path))
        print("rank=%d processed=%d/%d frames" % (args.rank, min(batch_start + batch_size, len(selected)), len(selected)), flush=True)

    annotations_handle.close()
    metrics_handle.close()
    summary = {
        "rank": int(args.rank),
        "world_size": int(args.world_size),
        "frames": len(selected),
        "annotations": annotation_count,
        "metrics": metric_count,
        "mask_match_count": mask_match_count,
        "mask_match_fraction": mask_match_count / annotation_count if annotation_count else 0.0,
        "detector_config": detector_config,
    }
    (shard_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    preview_root = shard_root / "visualizations"
    for record, detections, candidates, thermal_image, rgb_path in preview_records:
        render_preview(
            preview_root / (str(record["sequence"]) + "_" + str(record["frame_id"]) + ".jpg"),
            rgb_path,
            thermal_image,
            detections,
            candidates,
        )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def summarize(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    metric_rows = []
    for path in sorted((output_root / "shards").glob("rank*/metrics.jsonl")):
        metric_rows.extend(read_jsonl(path))
    annotation_methods = {"robust_depth_corners", "mask_depth_foreground"}
    summary: dict[str, Any] = {
        "metrics_are_proxies_not_gt": True,
        "frames": len({(row["sequence"], row["frame_id"]) for row in metric_rows}),
        "annotations": len(
            {
                row["source_line_index"]
                for row in metric_rows
                if row["method"] in annotation_methods
            }
        ),
        "methods": {},
        "per_class": {},
    }
    for method in METHODS:
        rows = [row for row in metric_rows if row["method"] == method]
        if not rows:
            continue
        summary["methods"][method] = {
            "boxes": len(rows),
            "edge_alignment_median": float(np.median([row["edge_alignment"] for row in rows])),
            "thermal_contrast_median": float(np.median([row["thermal_contrast"] for row in rows])),
        }
        for class_name in sorted({row["class_name"] for row in rows}):
            key = (class_name, method)
            subset = [row for row in rows if row["class_name"] == class_name]
            summary["per_class"]["%s/%s" % key] = {
                "boxes": len(subset),
                "edge_alignment_median": float(np.median([row["edge_alignment"] for row in subset])),
                "thermal_contrast_median": float(np.median([row["thermal_contrast"] for row in subset])),
            }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def merge(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    final_root = output_root / "final"
    if final_root.exists() and any(final_root.iterdir()):
        raise RuntimeError("final output exists and is not empty: %s" % final_root)
    final_root.mkdir(parents=True, exist_ok=True)
    shard_rows: dict[int, dict[str, Any]] = {}
    for path in sorted((output_root / "shards").glob("rank*/annotations.jsonl")):
        for row in read_jsonl(path):
            shard_rows[int(row["source_line_index"])] = row
    source_rows = list(read_jsonl(input_root / "annotations.jsonl"))
    if len(shard_rows) != len(source_rows):
        raise RuntimeError("full merge coverage mismatch: shards=%d source=%d" % (len(shard_rows), len(source_rows)))
    final_annotations = []
    for index, source in enumerate(source_rows):
        row = dict(shard_rows[index])
        row.pop("source_line_index", None)
        final_annotations.append(row)
    write_jsonl(final_annotations, final_root / "annotations.jsonl")

    frame_rows = list(read_jsonl(input_root / "frames.jsonl"))
    by_frame: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in final_annotations:
        by_frame[frame_key(row)].append(row)
    records = []
    for record in read_jsonl(input_root / "records.jsonl"):
        updated = dict(record)
        detections = by_frame.get(frame_key(record), [])
        updated["boxes"] = [row["bbox_thr_xyxy"] for row in detections]
        updated["annotation_source"] = "sam3.1_rgb_mask+ms2_calib+proj_depth_refined+thermal_bounded_refine"
        updated["transfer_method"] = "mask_depth_thermal_refine"
        records.append(updated)
    write_jsonl(records, final_root / "records.jsonl")
    write_jsonl(frame_rows, final_root / "frames.jsonl")
    for filename in ("manifest.jsonl", "rejections.jsonl"):
        source = input_root / filename
        if source.is_file():
            final_root.joinpath(filename).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    image_ids = {frame_key(row): index for index, row in enumerate(frame_rows, start=1)}
    category_ids = {name: index for index, name in enumerate(CATEGORY_NAMES, start=1)}
    images = [
        {
            "id": image_ids[frame_key(row)],
            "file_name": row["thermal_path"],
            "width": int(row["thermal_width"]),
            "height": int(row["thermal_height"]),
            "sequence": row["sequence"],
            "frame_id": row["frame_id"],
        }
        for row in frame_rows
    ]
    coco_annotations = []
    for annotation_id, row in enumerate(final_annotations, start=1):
        x1, y1, x2, y2 = (float(value) for value in row["bbox_thr_xyxy"])
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        coco_annotations.append({
            "id": annotation_id,
            "image_id": image_ids[frame_key(row)],
            "category_id": category_ids[row["class_name"]],
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0,
            "score": float(row["score"]),
            "source": row["source"],
            "bbox_rgb_xyxy": row["bbox_rgb_xyxy"],
            "mask_match_iou": row.get("mask_match_iou"),
            "mask_depth_valid_count": row.get("mask_depth_valid_count"),
        })
    (final_root / "instances.json").write_text(json.dumps({
        "images": images,
        "annotations": coco_annotations,
        "categories": [
            {"id": category_ids[name], "name": name, "supercategory": name.split(".", 1)[0]}
            for name in CATEGORY_NAMES
        ],
    }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    source_metadata = json.loads((input_root / "metadata.json").read_text(encoding="utf-8"))
    metadata = dict(source_metadata)
    metadata.pop("shards", None)
    metadata.update({
        "source": "SAM3.1 RGB instance mask + MS2 timestamp pairing + RGB refined depth + calibration projection + bounded thermal refinement",
        "parent_output": str(input_root),
        "transfer_method": "mask_depth_thermal_refine",
        "mask_inference": "SAM3.1 save_masks=True; masks matched to source RGB detections by class and bbox IoU",
        "manual_review_required": True,
        "derived_alternative": True,
        "detections": len(final_annotations),
        "filtered_detections": len(final_annotations),
        "raw_images_copied": False,
    })
    (final_root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    mask_matches = [row for row in final_annotations if row.get("mask_match_iou") is not None]
    report = {
        "status": "complete",
        "transfer_method": "mask_depth_thermal_refine",
        "frames_total": len(frame_rows),
        "frames_with_boxes": len(by_frame),
        "detections_total": len(final_annotations),
        "mask_match_fraction": len(mask_matches) / len(final_annotations) if final_annotations else 0.0,
        "mask_match_iou_median": float(np.median([row["mask_match_iou"] for row in mask_matches])) if mask_matches else None,
        "detections_by_class": dict(sorted(Counter(row["class_name"] for row in final_annotations).items())),
        "bbox_coordinate_system": "left thermal pixel xyxy; instances.json uses xywh",
        "manual_review_required": True,
        "raw_images_copied": False,
    }
    (final_root / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "summarize", "merge"), required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sam3-eval-root", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--frames-per-sequence", type=int, default=2)
    parser.add_argument("--max-boxes-per-frame", type=int, default=15)
    parser.add_argument("--preview-count", type=int, default=10)
    parser.add_argument("--preview-stride", type=int, default=1)
    parser.add_argument("--final-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-resolution", type=int, default=1008)
    parser.add_argument("--mask-resolution", type=int, default=256)
    parser.add_argument("--model-confidence", type=float, default=0.35)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--precision", default="bfloat16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.mode == "worker":
        worker(args)
    elif args.mode == "summarize":
        summarize(args)
    else:
        merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
