#!/usr/bin/env python3
"""Compare RGB-to-thermal bbox transfer strategies on MS2 samples.

The experiment uses the same high-confidence RGB detections for every method.
Thermal edge contrast and released thermal depth provide proxy metrics; they
are not substitutes for manually labeled MS2 thermal boxes.
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

from ms2_pseudo_bbox import load_calibration, load_depth, resolve_project_path


METHODS = (
    "depth_cloud",
    "image_scale",
    "rotation_only",
    "robust_depth_corners",
    "robust_center_focal_size",
)
METHOD_TITLES = {
    "depth_cloud": "A current depth-cloud",
    "image_scale": "B image-ratio scale",
    "rotation_only": "C rotation homography",
    "robust_depth_corners": "D robust-depth corners",
    "robust_center_focal_size": "E robust center + focal size",
}
COLORS = {
    "human.pedestrian": (0, 255, 0),
    "vehicle.car": (255, 80, 30),
    "vehicle.truck": (255, 210, 0),
    "vehicle.bus": (0, 220, 255),
    "vehicle.motorcycle": (255, 0, 255),
    "vehicle.bicycle": (80, 160, 255),
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def frame_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["sequence"]), str(row["frame_id"])


def select_frames(records_path: Path, frames_per_sequence: int) -> list[dict[str, Any]]:
    by_sequence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(records_path):
        if row.get("boxes"):
            by_sequence[str(row["sequence"])].append(row)
    selected: list[dict[str, Any]] = []
    for sequence, rows in sorted(by_sequence.items()):
        positions = np.linspace(
            0,
            len(rows) - 1,
            min(frames_per_sequence, len(rows)),
            dtype=np.int64,
        )
        selected.extend(rows[int(position)] for position in sorted(set(positions)))
    return selected


def valid_box(box: Sequence[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = max(0.0, min(float(width - 1), x1))
    x2 = max(0.0, min(float(width - 1), x2))
    y1 = max(0.0, min(float(height - 1), y1))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return [x1, y1, x2, y2]


def rgb_depth_quantile(
    depth: np.ndarray,
    bbox: Sequence[float],
    quantile: float = 25.0,
) -> tuple[float, int] | None:
    height, width = depth.shape
    x1, y1, x2, y2 = (float(value) for value in bbox)
    # Remove the outer 10% of the detection where background leakage is common.
    inset_x = 0.10 * max(0.0, x2 - x1)
    inset_y = 0.10 * max(0.0, y2 - y1)
    left = max(0, min(width - 1, int(math.floor(x1 + inset_x))))
    right = max(left + 1, min(width, int(math.ceil(x2 - inset_x))))
    top = max(0, min(height - 1, int(math.floor(y1 + inset_y))))
    bottom = max(top + 1, min(height, int(math.ceil(y2 - inset_y))))
    values = depth[top:bottom, left:right]
    valid = values[np.isfinite(values) & (values > 0.1) & (values < 200.0)]
    if valid.size < 24:
        return None
    return float(np.percentile(valid, quantile)), int(valid.size)


def project_pixels_at_depth(
    pixels: np.ndarray,
    depth_m: float,
    k_rgb: np.ndarray,
    k_thr: np.ndarray,
    rgb_to_thr: np.ndarray,
    include_translation: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    homogeneous = np.column_stack((pixels, np.ones(len(pixels), dtype=np.float64)))
    rays = homogeneous @ np.linalg.inv(k_rgb).T
    points_rgb_m = rays * float(depth_m)
    rotation = rgb_to_thr[:, :3]
    translation_m = rgb_to_thr[:, 3] / 1000.0 if include_translation else 0.0
    points_thr_m = points_rgb_m @ rotation.T + translation_m
    positive = points_thr_m[:, 2] > 0.1
    if not bool(positive.all()):
        return None
    projected_h = points_thr_m @ k_thr.T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    return projected, points_thr_m[:, 2]


def candidate_boxes(
    detection: dict[str, Any],
    depth_rgb: np.ndarray,
    k_rgb: np.ndarray,
    k_thr: np.ndarray,
    rgb_to_thr: np.ndarray,
    rgb_size: tuple[int, int],
    thermal_size: tuple[int, int],
) -> tuple[dict[str, list[float]], float | None, int]:
    rgb_width, rgb_height = rgb_size
    thermal_width, thermal_height = thermal_size
    rgb_box = [float(value) for value in detection["bbox_rgb_xyxy"]]
    corners = np.asarray(
        [
            [rgb_box[0], rgb_box[1]],
            [rgb_box[2], rgb_box[1]],
            [rgb_box[2], rgb_box[3]],
            [rgb_box[0], rgb_box[3]],
        ],
        dtype=np.float64,
    )
    center = np.asarray(
        [[0.5 * (rgb_box[0] + rgb_box[2]), 0.5 * (rgb_box[1] + rgb_box[3])]],
        dtype=np.float64,
    )
    result: dict[str, list[float]] = {}
    current = valid_box(detection["bbox_thr_xyxy"], thermal_width, thermal_height)
    if current is not None:
        result["depth_cloud"] = current
    scaled = valid_box(
        [
            rgb_box[0] * thermal_width / rgb_width,
            rgb_box[1] * thermal_height / rgb_height,
            rgb_box[2] * thermal_width / rgb_width,
            rgb_box[3] * thermal_height / rgb_height,
        ],
        thermal_width,
        thermal_height,
    )
    if scaled is not None:
        result["image_scale"] = scaled
    rotation_projection = project_pixels_at_depth(
        corners, 1000.0, k_rgb, k_thr, rgb_to_thr, include_translation=False
    )
    if rotation_projection is not None:
        uv, _ = rotation_projection
        box = valid_box(
            [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
            thermal_width,
            thermal_height,
        )
        if box is not None:
            result["rotation_only"] = box
    depth_result = rgb_depth_quantile(depth_rgb, rgb_box)
    if depth_result is None:
        return result, None, 0
    depth_m, support = depth_result
    corner_projection = project_pixels_at_depth(
        corners, depth_m, k_rgb, k_thr, rgb_to_thr, include_translation=True
    )
    center_projection = project_pixels_at_depth(
        center, depth_m, k_rgb, k_thr, rgb_to_thr, include_translation=True
    )
    expected_thermal_depth = None
    if corner_projection is not None:
        uv, depths = corner_projection
        expected_thermal_depth = float(np.median(depths))
        box = valid_box(
            [uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()],
            thermal_width,
            thermal_height,
        )
        if box is not None:
            result["robust_depth_corners"] = box
    if center_projection is not None:
        uv_center, depths = center_projection
        expected_thermal_depth = float(depths[0])
        half_width = 0.5 * (rgb_box[2] - rgb_box[0]) * k_thr[0, 0] / k_rgb[0, 0]
        half_height = 0.5 * (rgb_box[3] - rgb_box[1]) * k_thr[1, 1] / k_rgb[1, 1]
        cx, cy = uv_center[0]
        box = valid_box(
            [cx - half_width, cy - half_height, cx + half_width, cy + half_height],
            thermal_width,
            thermal_height,
        )
        if box is not None:
            result["robust_center_focal_size"] = box
    return result, expected_thermal_depth, support


def thermal_display(array: np.ndarray) -> tuple[Image.Image, np.ndarray, float]:
    finite = array[np.isfinite(array)]
    low, high = np.percentile(finite.astype(np.float32), [1.0, 99.0])
    scale = max(float(high - low), 1e-6)
    normalized = np.clip((array.astype(np.float32) - low) / scale, 0.0, 1.0)
    image = Image.fromarray((normalized * 255.0).astype(np.uint8), mode="L").convert("RGB")
    gy, gx = np.gradient(normalized)
    gradient = np.hypot(gx, gy)
    gradient_norm = max(float(np.percentile(gradient, 90.0)), 1e-6)
    return image, gradient, gradient_norm


def box_metrics(
    box: Sequence[float],
    thermal: np.ndarray,
    gradient: np.ndarray,
    gradient_norm: float,
    thermal_depth: np.ndarray | None,
    expected_depth_m: float | None,
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
    robust_range = max(float(np.percentile(thermal, 99) - np.percentile(thermal, 1)), 1e-6)
    contrast = (
        abs(float(np.median(inside)) - float(np.median(ring))) / robust_range
        if inside.size and ring.size
        else 0.0
    )
    result: dict[str, float | int | None] = {
        "edge_alignment": edge_alignment,
        "thermal_contrast": contrast,
        "depth_support": None,
        "depth_purity": None,
    }
    if thermal_depth is not None and expected_depth_m is not None:
        values = thermal_depth[top:bottom, left:right]
        valid = values[np.isfinite(values) & (values > 0.1) & (values < 200.0)]
        result["depth_support"] = int(valid.size)
        if valid.size:
            tolerance = max(0.75, 0.15 * expected_depth_m)
            result["depth_purity"] = float(
                np.mean(np.abs(valid - expected_depth_m) <= tolerance)
            )
    return result


def draw_boxes(
    base: Image.Image,
    detections: Sequence[dict[str, Any]],
    boxes_by_detection: Sequence[dict[str, list[float]]],
    method: str,
) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
    draw.text((4, 3), METHOD_TITLES[method], fill=(255, 255, 255))
    for detection, candidates in zip(detections, boxes_by_detection, strict=True):
        box = candidates.get(method)
        if box is None:
            continue
        color = COLORS.get(str(detection["class_name"]), (255, 255, 255))
        draw.rectangle(tuple(box), outline=color, width=2)
    return image


def draw_rgb(image_path: Path, detections: Sequence[dict[str, Any]]) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    image.thumbnail((640, 238), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (640, 256), "black")
    canvas.paste(image, (0, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), "RGB detections", fill=(255, 255, 255))
    scale_x = image.width / 1224.0
    scale_y = image.height / 384.0
    for detection in detections:
        box = detection["bbox_rgb_xyxy"]
        scaled = [box[0] * scale_x, box[1] * scale_y + 18, box[2] * scale_x, box[3] * scale_y + 18]
        color = COLORS.get(str(detection["class_name"]), (255, 255, 255))
        draw.rectangle(tuple(scaled), outline=color, width=2)
    return canvas


def run(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    selected_records = select_frames(
        input_root / "records.jsonl", args.frames_per_sequence
    )
    selected_keys = {frame_key(row) for row in selected_records}
    detections_by_key: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(input_root / "annotations.jsonl"):
        key = frame_key(row)
        if key in selected_keys:
            detections_by_key[key].append(row)
    (output_root / "visualizations").mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    visual_count = 0
    for frame_index, record in enumerate(selected_records):
        key = frame_key(record)
        detections = sorted(
            detections_by_key[key], key=lambda row: float(row["score"]), reverse=True
        )[: args.max_boxes_per_frame]
        if not detections:
            continue
        rgb_path = resolve_project_path(record["rgb_path"], project_root)
        thermal_path = resolve_project_path(record["image"], project_root)
        first_detection = detections[0]
        depth_rgb_path = resolve_project_path(first_detection["depth_rgb_path"], project_root)
        depth_thr_path = resolve_project_path(first_detection.get("depth_thr_path"), project_root)
        calib_path = resolve_project_path(first_detection["calib_path"], project_root)
        if rgb_path is None or thermal_path is None or depth_rgb_path is None or calib_path is None:
            continue
        depth_rgb, depth_scale = load_depth(depth_rgb_path)
        depth_rgb = depth_rgb * depth_scale
        thermal_depth = None
        if depth_thr_path is not None and depth_thr_path.is_file():
            thermal_depth, thermal_scale = load_depth(depth_thr_path)
            thermal_depth = thermal_depth * thermal_scale
        k_rgb, k_thr, rgb_to_thr = load_calibration(calib_path)
        with Image.open(thermal_path) as thermal_source:
            thermal_array = np.asarray(thermal_source).astype(np.float32)
        thermal_image, gradient, gradient_norm = thermal_display(thermal_array)
        boxes_by_detection: list[dict[str, list[float]]] = []
        for detection in detections:
            candidates, expected_depth, rgb_depth_support = candidate_boxes(
                detection,
                depth_rgb,
                k_rgb,
                k_thr,
                rgb_to_thr,
                (int(detection["rgb_width"]), int(detection["rgb_height"])),
                (int(detection["thermal_width"]), int(detection["thermal_height"])),
            )
            boxes_by_detection.append(candidates)
            for method, box in candidates.items():
                metrics = box_metrics(
                    box,
                    thermal_array,
                    gradient,
                    gradient_norm,
                    thermal_depth,
                    expected_depth,
                )
                metric_rows.append(
                    {
                        "sequence": record["sequence"],
                        "frame_id": record["frame_id"],
                        "class_name": detection["class_name"],
                        "score": float(detection["score"]),
                        "method": method,
                        "bbox_xyxy": box,
                        "expected_depth_m": expected_depth,
                        "rgb_depth_support": rgb_depth_support,
                        **metrics,
                    }
                )
        if frame_index % max(1, len(selected_records) // args.visual_count) == 0:
            panels = [draw_rgb(rgb_path, detections)] + [
                draw_boxes(thermal_image, detections, boxes_by_detection, method)
                for method in METHODS
            ]
            canvas = Image.new("RGB", (640 * 3, 256 * 2), "black")
            for index, panel in enumerate(panels):
                canvas.paste(panel, ((index % 3) * 640, (index // 3) * 256))
            canvas.save(
                output_root
                / "visualizations"
                / f"{record['sequence']}_{record['frame_id']}.jpg",
                quality=94,
            )
            visual_count += 1
    metric_path = output_root / "metrics.jsonl"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("w", encoding="utf-8") as handle:
        for row in metric_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    summary: dict[str, Any] = {
        "frames_selected": len(selected_records),
        "visualizations": visual_count,
        "metrics_are_proxies_not_gt": True,
        "methods": {},
        "per_class": {},
    }
    for method in METHODS:
        rows = [row for row in metric_rows if row["method"] == method]
        method_summary: dict[str, Any] = {"boxes": len(rows)}
        for metric in ("edge_alignment", "thermal_contrast", "depth_purity"):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            method_summary[metric] = {
                "median": float(np.median(values)) if values else None,
                "p75": float(np.percentile(values, 75)) if values else None,
                "mean": float(np.mean(values)) if values else None,
            }
        method_summary["depth_supported_fraction"] = float(
            np.mean([int(row["depth_support"] or 0) >= 8 for row in rows])
        ) if rows else 0.0
        summary["methods"][method] = method_summary
    classes = sorted({str(row["class_name"]) for row in metric_rows})
    for class_name in classes:
        summary["per_class"][class_name] = {}
        for method in METHODS:
            rows = [
                row
                for row in metric_rows
                if row["method"] == method and row["class_name"] == class_name
            ]
            summary["per_class"][class_name][method] = {
                "boxes": len(rows),
                "edge_median": float(np.median([row["edge_alignment"] for row in rows])) if rows else None,
                "contrast_median": float(np.median([row["thermal_contrast"] for row in rows])) if rows else None,
                "depth_purity_median": float(np.median([row["depth_purity"] for row in rows if row["depth_purity"] is not None]))
                if any(row["depth_purity"] is not None for row in rows)
                else None,
            }
    atomic_json_dump(summary, output_root / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=5)
    parser.add_argument("--max-boxes-per-frame", type=int, default=15)
    parser.add_argument("--visual-count", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
