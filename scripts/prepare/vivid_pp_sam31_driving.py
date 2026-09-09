#!/usr/bin/env python3
"""Generate pilot thermal bboxes for ViViD++ driving-full bags.

SAM3.1 runs only on the RGB transition image. LiDAR points from the paired
OS1 scan are projected through the official RGB/thermal calibration, and the
resulting boxes are written in thermal-image coordinates. This script is
non-destructive and never changes raw bags or existing annotations.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import vivid_pp_calibration_pilot as streams
import vivid_pp_lidar_calibration_pilot as lidar


PROMPTS = ("person", "car", "truck", "bus", "motorcycle", "bicycle")
CATEGORY_NAMES = (
    "human.pedestrian",
    "vehicle.bicycle",
    "vehicle.bus",
    "vehicle.car",
    "vehicle.motorcycle",
    "vehicle.truck",
)
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


@dataclass(frozen=True)
class ThermalProjection:
    bbox_xyxy: list[float]
    bbox_lidar_xyxy: list[float]
    bbox_geometry_xyxy: list[float]
    bbox_unexpanded_xyxy: list[float]
    lidar_point_count: int
    thermal_point_count: int
    rgb_in_frame_fraction: float
    thermal_in_frame_fraction: float
    point_selection_source: str
    thermal_margin_ratio: float


@dataclass(frozen=True)
class Candidate:
    class_name: str
    detector_label: str
    score: float
    bbox_rgb_xyxy: list[float]
    mask: np.ndarray | None

class PilotError(RuntimeError):
    """Raised when an annotation pilot precondition is not met."""


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_class(prompt: str) -> str | None:
    normalized = prompt.strip().lower()
    for label, class_name in PROMPT_TO_CLASS.items():
        if label in normalized:
            return class_name
    return None


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in first)
    bx1, by1, bx2, by2 = (float(value) for value in second)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def classwise_nms(candidates: list[Candidate], threshold: float) -> list[Candidate]:
    kept: list[Candidate] = []
    for class_name in sorted({candidate.class_name for candidate in candidates}):
        ordered = sorted(
            (candidate for candidate in candidates if candidate.class_name == class_name),
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        while ordered:
            current = ordered.pop(0)
            kept.append(current)
            ordered = [
                candidate
                for candidate in ordered
                if box_iou(current.bbox_rgb_xyxy, candidate.bbox_rgb_xyxy) < threshold
            ]
    return sorted(kept, key=lambda candidate: candidate.score, reverse=True)


def thermal_nms(detections: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for class_name in sorted({str(detection["class_name"]) for detection in detections}):
        ordered = sorted(
            (detection for detection in detections if detection["class_name"] == class_name),
            key=lambda detection: float(detection["score"]),
            reverse=True,
        )
        while ordered:
            current = ordered.pop(0)
            kept.append(current)
            ordered = [
                detection
                for detection in ordered
                if box_iou(current["bbox_thr_xyxy"], detection["bbox_thr_xyxy"]) < threshold
            ]
    return sorted(kept, key=lambda detection: float(detection["score"]), reverse=True)


def load_detector(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    sam3_eval_root = args.sam3_eval_root.resolve()
    if str(sam3_eval_root) not in sys.path:
        sys.path.insert(0, str(sam3_eval_root))
    from sam31_detector import Sam31Detector

    config = {
        "input_resolution": int(args.input_resolution),
        "mask_resolution": 256,
        "save_masks": True,
        "prompts": list(PROMPTS),
        "confidence_threshold": float(args.confidence_threshold),
        "max_detections_per_prompt": int(args.max_detections_per_prompt),
        "max_detections_per_image": int(args.max_detections_per_image),
        "nms_iou_threshold": float(args.nms_iou),
        "mask_threshold": 0.5,
        "precision": str(args.precision),
        "checkpoint_minimum_coverage": 0.95,
        "checkpoint_mmap": True,
    }
    detector = Sam31Detector(
        config=config,
        device=torch.device(args.device),
        sam3_repo=str(sam3_eval_root / "sam3"),
        checkpoint=str(args.checkpoint.resolve()),
    )
    return detector, torch, config


def align_detector_visual_devices(detector: Any) -> None:
    """Move SAM3 non-parameter tensors and caches onto the selected GPU.

    Parameters move with ``model.to(device)``, but this SAM3 fork creates
    positional tensors and decoder coordinate caches during construction.
    Those tensors default to cuda:0 and must be cleared or moved for cuda:1.
    """

    if getattr(detector.device, "type", None) != "cuda":
        return
    for module in detector.model.modules():
        if hasattr(module, "compilable_cord_cache"):
            module.compilable_cord_cache = None
            module.compilable_stored_size = None
            if hasattr(module, "coord_cache"):
                module.coord_cache.clear()
    original_run_prompt = detector.run_prompt

    def run_prompt(
        visual_outputs: dict[str, Any],
        metadata: list[dict[str, Any]],
        prompt_index: int,
        per_image_outputs: list[list[dict[str, Any]]],
    ) -> None:
        positional = visual_outputs.get("vision_pos_enc")
        if positional is not None:
            visual_outputs["vision_pos_enc"] = [
                value.to(detector.device) for value in positional
            ]
        original_run_prompt(visual_outputs, metadata, prompt_index, per_image_outputs)

    detector.run_prompt = run_prompt


def rgb_tensor(sample: streams.ImageSample, torch_module: Any) -> Any:
    array = sample.array
    if sample.encoding == "bgr8":
        array = array[..., ::-1]
    array = np.ascontiguousarray(array)
    if array.ndim != 3 or array.shape[2] != 3:
        raise PilotError(f"RGB sample is not 3-channel: {sample.encoding} {array.shape}")
    return torch_module.from_numpy(array).permute(2, 0, 1).contiguous()


def _inside_uv(uv: np.ndarray, positive: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        positive
        & np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )


def project_lidar(
    points: np.ndarray,
    stereo: streams.StereoCalibration,
    lidar_calibration: lidar.LidarCalibration,
    lidar_to_rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_rgb = lidar.apply_transform(points, lidar_to_rgb)
    rgb_uv, rgb_positive = lidar.project_points(
        points_rgb,
        lidar_calibration.K,
        lidar_calibration.distortion,
        lidar_calibration.distortion_model,
    )
    points_thermal = lidar.apply_transform(points_rgb, stereo.rgb_to_thermal_m)
    thermal_uv, thermal_positive = lidar.project_points(
        points_thermal,
        stereo.thermal.K,
        stereo.thermal.distortion,
        stereo.thermal.distortion_model,
    )
    rgb_inside = _inside_uv(
        rgb_uv, rgb_positive, lidar_calibration.resolution[0], lidar_calibration.resolution[1]
    )
    thermal_inside = _inside_uv(
        thermal_uv, thermal_positive, stereo.thermal.resolution[0], stereo.thermal.resolution[1]
    )
    return rgb_uv, thermal_uv, rgb_inside, thermal_inside, points_rgb


def _unproject_rgb_pixels(
    raw_uv: np.ndarray,
    depth_m: np.ndarray,
    calibration: lidar.LidarCalibration,
) -> np.ndarray:
    cv2 = streams._cv2()
    raw_points = np.asarray(raw_uv, dtype=np.float64).reshape(-1, 1, 2)
    if calibration.distortion_model == "equidistant":
        undistorted = cv2.fisheye.undistortPoints(
            raw_points,
            calibration.K,
            calibration.distortion.reshape(-1, 1),
            R=np.eye(3, dtype=np.float64),
            P=calibration.K,
        ).reshape(-1, 2)
    else:
        undistorted = cv2.undistortPoints(
            raw_points,
            calibration.K,
            calibration.distortion,
            R=np.eye(3, dtype=np.float64),
            P=calibration.K,
        ).reshape(-1, 2)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    return np.column_stack(
        (
            (undistorted[:, 0] - calibration.K[0, 2]) * depth_m / calibration.K[0, 0],
            (undistorted[:, 1] - calibration.K[1, 2]) * depth_m / calibration.K[1, 1],
            depth_m,
        )
    )


def _project_rgb_bbox_geometry(
    bbox_rgb_xyxy: Sequence[float],
    target_points_rgb: np.ndarray,
    stereo: streams.StereoCalibration,
    lidar_calibration: lidar.LidarCalibration,
    thermal_size: tuple[int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = (float(value) for value in bbox_rgb_xyxy)
    edge = np.linspace(0.0, 1.0, 24, dtype=np.float64)
    top = np.column_stack((x1 + (x2 - x1) * edge, np.full(edge.shape, y1)))
    bottom = np.column_stack((x1 + (x2 - x1) * edge, np.full(edge.shape, y2)))
    left = np.column_stack((np.full(edge.shape, x1), y1 + (y2 - y1) * edge))
    right = np.column_stack((np.full(edge.shape, x2), y1 + (y2 - y1) * edge))
    perimeter = np.concatenate((top, bottom, left, right), axis=0)
    depth_values = np.percentile(target_points_rgb[:, 2], [10.0, 50.0, 90.0])
    raw_uv = np.repeat(perimeter, len(depth_values), axis=0)
    depths = np.tile(depth_values, len(perimeter))
    points_rgb = _unproject_rgb_pixels(raw_uv, depths, lidar_calibration)
    points_thermal = (
        points_rgb @ stereo.rgb_to_thermal_m[:3, :3].T
        + stereo.rgb_to_thermal_m[:3, 3]
    )
    thermal_uv, thermal_positive = lidar.project_points(
        points_thermal,
        stereo.thermal.K,
        stereo.thermal.distortion,
        stereo.thermal.distortion_model,
    )
    inside = _inside_uv(
        thermal_uv,
        thermal_positive,
        thermal_size[0],
        thermal_size[1],
    )
    return thermal_uv[inside]


def _box_from_uv(
    points: np.ndarray,
    image_size: tuple[int, int],
    quantile: float,
) -> list[float]:
    if points.size == 0:
        raise PilotError("empty_thermal_projection")
    low = np.percentile(points, quantile, axis=0)
    high = np.percentile(points, 100.0 - quantile, axis=0)
    width, height = image_size
    return [
        float(np.clip(low[0], 0.0, width - 1.0)),
        float(np.clip(low[1], 0.0, height - 1.0)),
        float(np.clip(high[0], 0.0, width - 1.0)),
        float(np.clip(high[1], 0.0, height - 1.0)),
    ]


def transfer_candidate(
    candidate: Candidate,
    rgb_uv: np.ndarray,
    thermal_uv: np.ndarray,
    rgb_inside: np.ndarray,
    thermal_inside: np.ndarray,
    points_rgb: np.ndarray,
    stereo: streams.StereoCalibration,
    lidar_calibration: lidar.LidarCalibration,
    rgb_size: tuple[int, int],
    thermal_size: tuple[int, int],
    min_lidar_points: int,
    min_thermal_height: float,
    min_thermal_area: float,
    thermal_margin_ratio: float,
) -> ThermalProjection:
    x1, y1, x2, y2 = candidate.bbox_rgb_xyxy
    in_rgb_box = (
        rgb_inside
        & (rgb_uv[:, 0] >= x1)
        & (rgb_uv[:, 0] <= x2)
        & (rgb_uv[:, 1] >= y1)
        & (rgb_uv[:, 1] <= y2)
    )
    point_selection_source = "rgb_bbox"
    if candidate.mask is not None:
        mask = np.asarray(candidate.mask, dtype=bool)
        if mask.ndim != 2:
            raise PilotError(f"invalid SAM mask shape:{mask.shape}")
        mask_height, mask_width = mask.shape
        rgb_width, rgb_height = rgb_size
        mask_selected = np.zeros(in_rgb_box.shape, dtype=bool)
        selected_indices = np.flatnonzero(in_rgb_box)
        if selected_indices.size:
            mask_columns = np.floor(
                rgb_uv[selected_indices, 0] * mask_width / rgb_width
            ).astype(np.int64)
            mask_rows = np.floor(
                rgb_uv[selected_indices, 1] * mask_height / rgb_height
            ).astype(np.int64)
            valid = (
                (mask_columns >= 0)
                & (mask_columns < mask_width)
                & (mask_rows >= 0)
                & (mask_rows < mask_height)
            )
            valid_indices = selected_indices[valid]
            mask_selected[valid_indices] = mask[
                mask_rows[valid], mask_columns[valid]
            ]
        if int(mask_selected.sum()) >= min_lidar_points:
            in_rgb_box &= mask_selected
            point_selection_source = "sam_mask"
        else:
            point_selection_source = "rgb_bbox_fallback"
    lidar_count = int(in_rgb_box.sum())
    in_thermal = in_rgb_box & thermal_inside
    thermal_points = thermal_uv[in_thermal]
    if lidar_count < min_lidar_points:
        raise PilotError(f"insufficient_lidar_points:{lidar_count}")
    if thermal_points.shape[0] < min_lidar_points:
        raise PilotError(f"insufficient_thermal_points:{thermal_points.shape[0]}")
    target_points_rgb = points_rgb[in_rgb_box]
    lidar_box = _box_from_uv(thermal_points, thermal_size, 2.0)
    geometry_points = _project_rgb_bbox_geometry(
        candidate.bbox_rgb_xyxy,
        target_points_rgb,
        stereo,
        lidar_calibration,
        thermal_size,
    )
    geometry_box = (
        _box_from_uv(geometry_points, thermal_size, 0.0)
        if geometry_points.size
        else list(lidar_box)
    )
    combined_points = (
        np.concatenate((thermal_points, geometry_points), axis=0)
        if geometry_points.size
        else thermal_points
    )
    unexpanded_box = _box_from_uv(combined_points, thermal_size, 0.5)
    unexpanded_width = unexpanded_box[2] - unexpanded_box[0]
    unexpanded_height = unexpanded_box[3] - unexpanded_box[1]
    margin_x = max(2.0, thermal_margin_ratio * unexpanded_width)
    margin_y = max(2.0, thermal_margin_ratio * unexpanded_height)
    width, height = thermal_size
    final_box = [
        float(np.clip(unexpanded_box[0] - margin_x, 0.0, width - 1.0)),
        float(np.clip(unexpanded_box[1] - margin_y, 0.0, height - 1.0)),
        float(np.clip(unexpanded_box[2] + margin_x, 0.0, width - 1.0)),
        float(np.clip(unexpanded_box[3] + margin_y, 0.0, height - 1.0)),
    ]
    thermal_width = final_box[2] - final_box[0]
    thermal_height = final_box[3] - final_box[1]
    if thermal_height < min_thermal_height:
        raise PilotError(f"thermal_box_height:{thermal_height:.3f}")
    if thermal_width * thermal_height < min_thermal_area:
        raise PilotError(f"thermal_box_area:{thermal_width * thermal_height:.3f}")
    return ThermalProjection(
        bbox_xyxy=final_box,
        bbox_lidar_xyxy=lidar_box,
        bbox_geometry_xyxy=geometry_box,
        bbox_unexpanded_xyxy=unexpanded_box,
        lidar_point_count=lidar_count,
        thermal_point_count=int(thermal_points.shape[0]),
        rgb_in_frame_fraction=float(rgb_inside.mean()),
        thermal_in_frame_fraction=float(thermal_points.shape[0] / max(lidar_count, 1)),
        point_selection_source=point_selection_source,
        thermal_margin_ratio=thermal_margin_ratio,
    )


def _save_rgb_thermal_preview(
    rgb: np.ndarray,
    rgb_encoding: str,
    thermal: np.ndarray,
    rgb_detections: list[dict[str, Any]],
    output_path: Path,
    modality: str,
) -> None:
    cv2 = streams._cv2()
    if rgb_encoding == "bgr8":
        rgb = rgb[..., ::-1]
    rgb_image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
    thermal_bgr = streams._thermal_preview(thermal)
    thermal_rgb = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2RGB)
    thermal_image = Image.fromarray(thermal_rgb).convert("RGB")
    image = rgb_image if modality == "rgb" else thermal_image
    draw = ImageDraw.Draw(image)
    for detection in rgb_detections:
        box = detection["bbox_rgb_xyxy"] if modality == "rgb" else detection["bbox_thr_xyxy"]
        color = COLORS.get(str(detection["class_name"]), (255, 255, 0))
        draw.rectangle(tuple(float(value) for value in box), outline=color, width=3)
        draw.text(
            (float(box[0]) + 2, float(box[1]) + 2),
            f"{detection['class_name']} {float(detection['score']):.2f}",
            fill=color,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=95)


def write_coco(output_root: Path, frame_rows: list[dict[str, Any]], detections: list[dict[str, Any]]) -> None:
    category_ids = {name: index for index, name in enumerate(CATEGORY_NAMES, start=1)}
    images = []
    for index, row in enumerate(frame_rows, start=1):
        row["coco_image_id"] = index
        images.append(
            {
                "id": index,
                "file_name": row["thermal_path"],
                "width": row["thermal_width"],
                "height": row["thermal_height"],
            }
        )
    image_ids = {(str(row["bag"]), float(row["thermal_stamp"])): row["coco_image_id"] for row in frame_rows}
    annotations = []
    for index, detection in enumerate(detections, start=1):
        image_id = image_ids[(str(detection["bag"]), float(detection["thermal_stamp"]))]
        x1, y1, x2, y2 = detection["bbox_thr_xyxy"]
        annotations.append(
            {
                "id": index,
                "image_id": image_id,
                "category_id": category_ids[detection["class_name"]],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "score": detection["score"],
            }
        )
    (output_root / "thermal_coco.json").write_text(
        json.dumps(
            {
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": index, "name": name} for name, index in category_ids.items()
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive for this non-production pilot")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    stereo = streams.load_stereo_calibration(args.calibration_root.resolve(), "driving")
    lidar_calibration = lidar.load_lidar_calibration(args.calibration_root.resolve(), "driving")
    detector, torch_module, detector_config = load_detector(args)
    align_detector_visual_devices(detector)
    frame_rows: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for bag in args.bag:
        images, clouds, collection = lidar.collect_streams(
            bag.resolve(), "driving", args.source_cap, args.lidar_topic or "/os1_cloud_node/points"
        )
        rgb_topic, thermal_topic = "/camera/image_color", "/thermal/image_raw"
        pairs = streams.pair_samples(
            images,
            "driving",
            args.max_sync_delta_ms,
            args.max_sync_delta_ms,
            0,
        )
        matches = lidar.select_cloud_pairs(
            pairs,
            clouds,
            lidar_calibration.dt_lidar_to_ros,
            args.max_lidar_delta_ms,
            args.limit,
        )
        if not matches:
            raise PilotError(f"no synchronized RGB/thermal/LiDAR samples in {bag}")
        stored = lidar_calibration.lidar_to_rgb
        # The two platform YAML conventions are intentionally explicit.
        lidar_to_rgb = streams._invert_transform(stored)
        runtime_rows: list[dict[str, Any]] = []
        tensors: list[Any] = []
        for pair, cloud, cloud_delta_ms in matches:
            tensors.append(rgb_tensor(pair.rgb, torch_module))
            runtime_rows.append(
                {
                    "pair": pair,
                    "cloud": cloud,
                    "cloud_delta_ms": cloud_delta_ms,
                    "collection": collection,
                }
            )
        for start in range(0, len(tensors), args.batch_size):
            batch_tensors = tensors[start : start + args.batch_size]
            batch_rows = runtime_rows[start : start + args.batch_size]
            metadata = [
                {"height": row["pair"].rgb.height, "width": row["pair"].rgb.width}
                for row in batch_rows
            ]
            model_outputs = detector.predict_batch(batch_tensors, metadata)
            for runtime, model_detections in zip(batch_rows, model_outputs, strict=True):
                pair = runtime["pair"]
                cloud = runtime["cloud"]
                frame_base = {
                    "bag": str(bag.resolve()),
                    "rgb_stamp": pair.rgb.stamp,
                    "thermal_stamp": pair.thermal.stamp,
                    "lidar_stamp_raw": cloud.stamp,
                    "lidar_stamp_ros": cloud.stamp + float(lidar_calibration.dt_lidar_to_ros or 0.0),
                    "rgb_thermal_delta_ms": pair.delta_ms,
                    "lidar_rgb_delta_ms": runtime["cloud_delta_ms"],
                    "rgb_width": pair.rgb.width,
                    "rgb_height": pair.rgb.height,
                    "thermal_width": pair.thermal.width,
                    "thermal_height": pair.thermal.height,
                    "rgb_sequence": pair.rgb.sequence,
                    "thermal_sequence": pair.thermal.sequence,
                    "status": "detector_empty",
                    "candidate_count": 0,
                    "accepted_count": 0,
                }
                frame_detections: list[dict[str, Any]] = []
                candidates: list[Candidate] = []
                for raw in model_detections:
                    class_name = canonical_class(str(raw["prompt"]))
                    if class_name is None:
                        rejections.append({**frame_base, "reason": "unsupported_prompt", "prompt": str(raw["prompt"])})
                        continue
                    candidates.append(
                        Candidate(
                            class_name=class_name,
                            detector_label=str(raw["prompt"]),
                            score=float(raw["score"]),
                            bbox_rgb_xyxy=[float(value) for value in raw["bbox_xyxy"]],
                            mask=(
                                np.asarray(raw["mask"], dtype=bool)
                                if raw.get("mask") is not None
                                else None
                            ),
                        )
                    )
                candidates = classwise_nms(candidates, args.nms_iou)
                frame_base["candidate_count"] = len(candidates)
                if candidates:
                    (
                        rgb_uv,
                        thermal_uv,
                        rgb_inside,
                        thermal_inside,
                        points_rgb,
                    ) = project_lidar(
                        cloud.points, stereo, lidar_calibration, lidar_to_rgb
                    )
                    for candidate in candidates:
                        try:
                            projection = transfer_candidate(
                                candidate,
                                rgb_uv,
                                thermal_uv,
                                rgb_inside,
                                thermal_inside,
                                points_rgb,
                                stereo,
                                lidar_calibration,
                                (pair.rgb.width, pair.rgb.height),
                                stereo.thermal.resolution,
                                args.min_lidar_points,
                                args.min_thermal_height,
                                args.min_thermal_area,
                                args.thermal_margin_ratio,
                            )
                        except PilotError as error:
                            rejections.append(
                                {
                                    **frame_base,
                                    "class_name": candidate.class_name,
                                    "score": candidate.score,
                                    "bbox_rgb_xyxy": candidate.bbox_rgb_xyxy,
                                    "reason": str(error),
                                }
                            )
                            continue
                        frame_detections.append(
                            {
                                **frame_base,
                                "class_name": candidate.class_name,
                                "detector_label": candidate.detector_label,
                                "score": candidate.score,
                                "bbox_rgb_xyxy": candidate.bbox_rgb_xyxy,
                                "bbox_thr_xyxy": projection.bbox_xyxy,
                                "bbox_thr_lidar_xyxy": projection.bbox_lidar_xyxy,
                                "bbox_thr_geometry_xyxy": projection.bbox_geometry_xyxy,
                                "bbox_thr_unexpanded_xyxy": projection.bbox_unexpanded_xyxy,
                                "lidar_point_count_rgb_bbox": projection.lidar_point_count,
                                "lidar_point_count_thermal": projection.thermal_point_count,
                                "rgb_in_frame_fraction": projection.rgb_in_frame_fraction,
                                "thermal_in_frame_fraction": projection.thermal_in_frame_fraction,
                                "point_selection_source": projection.point_selection_source,
                                "thermal_margin_ratio": projection.thermal_margin_ratio,
                                "source": "sam3.1_rgb+vivid_driving_lidar+official_rgbthermal_calibration",
                                "target_modality": "thermal",
                            }
                        )
                    frame_detections = thermal_nms(
                        frame_detections, args.thermal_nms_iou
                    )
                    frame_base["accepted_count"] = len(frame_detections)
                    frame_base["status"] = "accepted" if frame_detections else "geometry_rejected"
                status_counts[frame_base["status"]] += 1
                rgb_path = output_root / "images" / "rgb" / f"{len(frame_rows):06d}.png"
                thermal_path = output_root / "images" / "thermal" / f"{len(frame_rows):06d}.png"
                streams._save_image(pair.rgb.array, rgb_path, pair.rgb.encoding)
                streams._save_image(pair.thermal.array, thermal_path, pair.thermal.encoding)
                frame_base["rgb_path"] = str(rgb_path.relative_to(output_root))
                frame_base["thermal_path"] = str(thermal_path.relative_to(output_root))
                frame_rows.append(frame_base)
                for detection in frame_detections:
                    detection["rgb_path"] = frame_base["rgb_path"]
                    detection["thermal_path"] = frame_base["thermal_path"]
                    detections.append(detection)
                if args.preview_count > 0 and len(frame_rows) <= args.preview_count:
                    _save_rgb_thermal_preview(
                        pair.rgb.array,
                        pair.rgb.encoding,
                        pair.thermal.array,
                        frame_detections,
                        output_root / "previews" / "rgb" / f"{len(frame_rows) - 1:06d}.jpg",
                        "rgb",
                    )
                    _save_rgb_thermal_preview(
                        pair.rgb.array,
                        pair.rgb.encoding,
                        pair.thermal.array,
                        frame_detections,
                        output_root / "previews" / "thermal" / f"{len(frame_rows) - 1:06d}.jpg",
                        "thermal",
                    )
        print(
            f"{bag.name}: pairs={len(matches)} statuses={dict(status_counts)} "
            f"boxes={len(detections)}",
            flush=True,
        )
    frames_path = output_root / "frames.jsonl"
    detections_path = output_root / "detections.jsonl"
    rejections_path = output_root / "rejections.jsonl"
    for path, rows in (
        (frames_path, frame_rows),
        (detections_path, detections),
        (rejections_path, rejections),
    ):
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_coco(output_root, frame_rows, detections)
    metadata = {
        "backend": "sam3.1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint.resolve()),
        "sam3_eval_root": str(args.sam3_eval_root.resolve()),
        "detector_config": detector_config,
        "thermal_nms_iou": float(args.thermal_nms_iou),
        "thermal_margin_ratio": float(args.thermal_margin_ratio),
        "bbox_transfer_contract": (
            "thermal bbox is the union of LiDAR-projected target points and "
            "RGB detector box perimeter projected at robust LiDAR depths, "
            "then expanded by the configured margin"
        ),
        "prompts": list(PROMPTS),
        "save_masks": True,
        "mask_guided_lidar_transfer": True,
        "target_modality": "thermal",
        "rgb_role": "transition_only",
        "calibration_source": stereo.source_path,
        "lidar_calibration_source": lidar_calibration.source_path,
        "lidar_transform_key": lidar_calibration.transform_key,
        "lidar_direction": "inverse" if args.mode == "driving" else "as_stored",
        "lidar_clock_offset_s": lidar_calibration.dt_lidar_to_ros,
        "frames": len(frame_rows),
        "detections": len(detections),
        "rejections": len(rejections),
        "status_counts": dict(status_counts),
        "manual_review_required": True,
        "production_rollout_allowed": False,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bag", type=Path, action="append", required=True)
    parser.add_argument("--sam3-eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("driving",), default="driving")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--input-resolution", type=int, default=1008)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--max-detections-per-prompt", type=int, default=100)
    parser.add_argument("--max-detections-per-image", type=int, default=150)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--thermal-nms-iou", type=float, default=0.50)
    parser.add_argument("--thermal-margin-ratio", type=float, default=0.10)
    parser.add_argument("--source-cap", type=int, default=60)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-sync-delta-ms", type=float, default=50.0)
    parser.add_argument("--max-lidar-delta-ms", type=float, default=50.0)
    parser.add_argument("--lidar-topic")
    parser.add_argument("--min-lidar-points", type=int, default=8)
    parser.add_argument("--min-thermal-height", type=float, default=8.0)
    parser.add_argument("--min-thermal-area", type=float, default=64.0)
    parser.add_argument("--preview-count", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if (
        args.source_cap <= 0
        or args.limit <= 0
        or args.batch_size <= 0
        or args.thermal_margin_ratio < 0.0
    ):
        raise SystemExit(
            "source-cap, limit, and batch-size must be positive; "
            "thermal-margin-ratio must be nonnegative"
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
