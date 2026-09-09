#!/usr/bin/env python3
"""Resumable, sharded SAM3.1 full annotation for MS2.

SAM3.1 detects in left RGB frames. RGB refined depth and MS2 calibration
project each detection into left thermal coordinates. Raw 16-bit thermal PNGs
are never copied or rewritten; COCO and JSONL records point to their paths.
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
from PIL import Image

from ms2_pseudo_bbox import (
    GeometryFailure,
    Sample,
    atomic_json_dump,
    atomic_jsonl_dump,
    canonical_class,
    classwise_nms,
    find_depth_file,
    image_files,
    load_calibration,
    load_depth,
    nearest_timestamp_index,
    read_timestamps,
    relative_to_project,
    resolve_project_path,
    save_visualization,
    timestamp_delta_ms,
    transfer_rgb_box_to_thermal,
)


PROMPTS = ("person", "car", "truck", "bus", "motorcycle", "bicycle")
CATEGORIES = (
    "human.pedestrian",
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus",
    "vehicle.motorcycle",
    "vehicle.bicycle",
)
CATEGORY_IDS = {name: index for index, name in enumerate(CATEGORIES, start=1)}
EXPECTED_CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"


def _write_jsonl_line(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=_json_default))
    handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _frame_key(row: dict[str, Any]) -> str:
    return f"{row['sequence']}:{row['frame_id']}"


def _timestamp_bad(row: dict[str, Any], max_delta_ms: float) -> bool:
    delta = row.get("timestamp_delta_ms")
    return delta is not None and abs(float(delta)) > max_delta_ms


def _direct_depth_file(root: Path, sequence: str, sensor: str, frame_id: str) -> Path | None:
    refined = root / "proj_depth_refined" / sequence / sensor / "depth_refined" / f"{frame_id}.npy"
    if refined.is_file():
        return refined
    return find_depth_file(root, sequence, sensor, frame_id)


def build_full_manifest(dataset_root: Path, project_root: Path) -> list[dict[str, Any]]:
    sync_root = dataset_root / "sync_data"
    if not sync_root.is_dir():
        raise FileNotFoundError(sync_root)
    rows: list[dict[str, Any]] = []
    for sequence_dir in sorted(item for item in sync_root.iterdir() if item.is_dir()):
        sequence = sequence_dir.name
        rgb_images = image_files(sequence_dir / "rgb" / "img_left")
        thermal_images = image_files(sequence_dir / "thr" / "img_left")
        if not rgb_images or not thermal_images:
            raise RuntimeError(f"missing RGB/thermal left frames for {sequence}")
        rgb_timestamps = read_timestamps(sequence_dir / "rgb" / "img_left_timestamp.txt")
        thermal_timestamps = read_timestamps(sequence_dir / "thr" / "img_left_timestamp.txt")
        if rgb_timestamps is None or thermal_timestamps is None:
            raise RuntimeError(f"missing timestamp stream for {sequence}")
        if len(rgb_images) != len(rgb_timestamps) or len(thermal_images) != len(thermal_timestamps):
            raise RuntimeError(f"timestamp/image count mismatch for {sequence}")
        with Image.open(rgb_images[0]) as rgb_sample, Image.open(thermal_images[0]) as thermal_sample:
            rgb_width, rgb_height = rgb_sample.size
            thermal_width, thermal_height = thermal_sample.size
        calib_path = sequence_dir / "calib.npy"
        if not calib_path.is_file():
            raise RuntimeError(f"missing calibration for {sequence}")
        for rgb_index, rgb_path in enumerate(rgb_images):
            rgb_timestamp = rgb_timestamps[rgb_index]
            thermal_index = nearest_timestamp_index(
                thermal_timestamps,
                rgb_timestamp,
                rgb_index,
            )
            thermal_path = thermal_images[thermal_index]
            frame_id = rgb_path.stem
            thermal_frame_id = thermal_path.stem
            rows.append(
                {
                    "sequence": sequence,
                    "frame_id": frame_id,
                    "rgb_path": relative_to_project(rgb_path, project_root),
                    "thermal_path": relative_to_project(thermal_path, project_root),
                    "depth_rgb_path": (
                        relative_to_project(depth_path, project_root)
                        if (depth_path := _direct_depth_file(dataset_root, sequence, "rgb", frame_id))
                        else None
                    ),
                    "depth_thr_path": (
                        relative_to_project(depth_path, project_root)
                        if (depth_path := _direct_depth_file(dataset_root, sequence, "thr", thermal_frame_id))
                        else None
                    ),
                    "calib_path": relative_to_project(calib_path, project_root),
                    "rgb_index": rgb_index,
                    "thermal_index": thermal_index,
                    "timestamp_rgb": rgb_timestamp,
                    "timestamp_thr": thermal_timestamps[thermal_index],
                    "timestamp_delta_ms": timestamp_delta_ms(
                        rgb_timestamp - thermal_timestamps[thermal_index]
                    ),
                    "rgb_width": rgb_width,
                    "rgb_height": rgb_height,
                    "thermal_width": thermal_width,
                    "thermal_height": thermal_height,
                }
            )
    counts = Counter(str(row["sequence"]) for row in rows)
    assignment_totals = [0, 0]
    assignments: dict[str, int] = {}
    for sequence, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        shard = 0 if assignment_totals[0] <= assignment_totals[1] else 1
        assignments[sequence] = shard
        assignment_totals[shard] += count
    for row in rows:
        row["shard"] = assignments[str(row["sequence"])]
    return rows


def command_manifest(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = build_full_manifest(dataset_root, project_root)
    atomic_jsonl_dump(rows, output_root / "manifest.jsonl")
    for shard in (0, 1):
        atomic_jsonl_dump(
            (row for row in rows if int(row["shard"]) == shard),
            output_root / f"manifest_gpu{shard}.jsonl",
        )
    counts = Counter(str(row["sequence"]) for row in rows)
    shard_counts = Counter(int(row["shard"]) for row in rows)
    atomic_json_dump(
        {
            "frames": len(rows),
            "sequences": len(counts),
            "frames_by_sequence": dict(sorted(counts.items())),
            "frames_by_shard": {str(key): value for key, value in sorted(shard_counts.items())},
            "timestamp_source": "MS2 sensor timestamp nearest-neighbor pairing",
            "raw_image_contract": "thermal_path points to original 16-bit PNG; no image copies",
        },
        output_root / "manifest_metadata.json",
    )
    print(f"wrote {len(rows)} frames across {len(counts)} sequences")
    print(f"shards: {dict(sorted(shard_counts.items()))}")


def load_detector(args: argparse.Namespace):
    import torch

    sam3_eval_root = Path(args.sam3_eval_root).resolve()
    sys.path.insert(0, str(sam3_eval_root))
    from sam31_detector import Sam31Detector

    config = {
        "input_resolution": args.input_resolution,
        "mask_resolution": 256,
        "save_masks": False,
        "prompts": list(PROMPTS),
        "confidence_threshold": args.confidence_threshold,
        "max_detections_per_prompt": 100,
        "max_detections_per_image": 150,
        "nms_iou_threshold": args.nms_iou,
        "mask_threshold": 0.5,
        "precision": args.precision,
        "checkpoint_minimum_coverage": 0.95,
        "checkpoint_mmap": True,
    }
    detector = Sam31Detector(
        config=config,
        device=torch.device(args.device),
        sam3_repo=str(sam3_eval_root / "sam3"),
        checkpoint=str(Path(args.checkpoint).resolve()),
    )
    return detector, torch, config


def _processed_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    keys.add(_frame_key(json.loads(line)))
                except json.JSONDecodeError:
                    # A killed process can leave one incomplete final line.
                    continue
    return keys


def _frame_rejection(row: dict[str, Any], reason: str, **extra: Any) -> dict[str, Any]:
    return {**row, "reason": reason, **extra}


def process_frame(
    row: dict[str, Any],
    model_detections: list[dict[str, Any]],
    project_root: Path,
    args: argparse.Namespace,
    calibration_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    frame = {**row, "status": "detector_empty", "model_candidate_count": 0, "candidate_count": 0, "accepted_count": 0}
    rejections: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    if _timestamp_bad(row, args.max_timestamp_delta_ms):
        frame["status"] = "timestamp_rejected"
        rejections.append(_frame_rejection(row, "timestamp_delta_exceeds_qc"))
        return frame, accepted, rejections

    candidates: list[dict[str, Any]] = []
    for detection in model_detections:
        class_name = canonical_class(str(detection["prompt"]))
        if class_name is None:
            rejections.append(
                _frame_rejection(
                    row,
                    "unsupported_prompt",
                    prompt=str(detection["prompt"]),
                    score=float(detection["score"]),
                )
            )
            continue
        candidates.append(
            {
                "class_name": class_name,
                "detector_label": str(detection["prompt"]),
                "score": float(detection["score"]),
                "bbox_rgb_xyxy": [float(value) for value in detection["bbox_xyxy"]],
            }
        )
    candidates = classwise_nms(candidates, args.nms_iou)
    frame["model_candidate_count"] = len(model_detections)
    frame["candidate_count"] = len(candidates)
    if not candidates:
        return frame, accepted, rejections

    depth_rgb_path = resolve_project_path(row.get("depth_rgb_path"), project_root)
    depth_thr_path = resolve_project_path(row.get("depth_thr_path"), project_root)
    calib_path = resolve_project_path(row["calib_path"], project_root)
    if depth_rgb_path is None or not depth_rgb_path.is_file():
        frame["status"] = "depth_rejected"
        rejections.extend(_frame_rejection(row, "missing_rgb_depth") for _ in candidates)
        return frame, accepted, rejections
    if calib_path is None or not calib_path.is_file():
        frame["status"] = "calibration_rejected"
        rejections.extend(_frame_rejection(row, "missing_calibration") for _ in candidates)
        return frame, accepted, rejections

    depth_rgb, depth_scale = load_depth(depth_rgb_path)
    if calib_path not in calibration_cache:
        calibration_cache[calib_path] = load_calibration(calib_path)
    k_rgb, k_thr, rgb_to_thr = calibration_cache[calib_path]
    depth_thr = None
    thermal_depth_scale = 1.0
    if depth_thr_path is not None and depth_thr_path.is_file():
        depth_thr, thermal_depth_scale = load_depth(depth_thr_path)

    for candidate in candidates:
        record = {**row, **candidate}
        try:
            projected = transfer_rgb_box_to_thermal(
                candidate["bbox_rgb_xyxy"],
                depth_rgb,
                depth_scale,
                k_rgb,
                k_thr,
                rgb_to_thr,
                (int(row["rgb_width"]), int(row["rgb_height"])),
                (int(row["thermal_width"]), int(row["thermal_height"])),
            )
            tx1, ty1, tx2, ty2 = projected["bbox_thr_xyxy"]
            width = float(tx2 - tx1)
            height = float(ty2 - ty1)
            area = width * height
            if height < args.min_thermal_height or area < args.min_thermal_area:
                raise GeometryFailure(
                    "thermal_box_below_quality_gate",
                    {
                        "thermal_box_width": width,
                        "thermal_box_height": height,
                        "thermal_box_area": area,
                    },
                )
            record.update(projected)
            record["bbox_thr_xyxy"] = projected.pop("bbox_thr_xyxy")
            record["source"] = "sam3.1_rgb+ms2_timestamp+calib+proj_depth_refined"
            if depth_thr is not None:
                x0 = max(0, int(math.floor(tx1)))
                x1 = min(depth_thr.shape[1], int(math.ceil(tx2)))
                y0 = max(0, int(math.floor(ty1)))
                y1 = min(depth_thr.shape[0], int(math.ceil(ty2)))
                if x1 > x0 and y1 > y0:
                    values = depth_thr[y0:y1, x0:x1] * thermal_depth_scale
                    valid = values[np.isfinite(values) & (values > 0.1)]
                    record["thermal_box_depth_valid_count"] = int(valid.size)
                    record["thermal_box_depth_median_m"] = float(np.median(valid)) if valid.size else None
            thermal_support = int(record.get("thermal_box_depth_valid_count", 0))
            if thermal_support < args.min_thermal_depth_support:
                raise GeometryFailure(
                    "thermal_depth_support_below_qc",
                    {"thermal_box_depth_valid_count": thermal_support},
                )
            accepted.append(record)
        except GeometryFailure as failure:
            rejections.append(_frame_rejection(row, failure.reason, **failure.diagnostics, **candidate))
    frame["accepted_count"] = len(accepted)
    frame["status"] = "accepted" if accepted else "geometry_rejected"
    return frame, accepted, rejections


def command_infer(args: argparse.Namespace) -> None:
    import torch

    manifest_path = Path(args.manifest).resolve()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows = list(_read_jsonl(manifest_path))
    if args.limit > 0:
        rows = rows[: args.limit]
    frame_path = output_root / "frames.jsonl"
    detection_path = output_root / "detections.jsonl"
    rejection_path = output_root / "rejections.jsonl"
    processed = _processed_keys(frame_path) if args.resume else set()
    if not args.resume:
        for path in (frame_path, detection_path, rejection_path):
            if path.exists():
                path.unlink()
    detector, torch_module, detector_config = load_detector(args)
    calibration_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    atomic_json_dump(
        {
            "backend": "sam3.1",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256_expected": EXPECTED_CHECKPOINT_SHA256,
            "device": args.device,
            "batch_size": args.batch_size,
            "input_resolution": args.input_resolution,
            "precision": args.precision,
            "prompts": list(PROMPTS),
            "confidence_threshold": args.confidence_threshold,
            "nms_iou": args.nms_iou,
            "max_timestamp_delta_ms": args.max_timestamp_delta_ms,
            "min_thermal_depth_support": args.min_thermal_depth_support,
            "min_thermal_height": args.min_thermal_height,
            "min_thermal_area": args.min_thermal_area,
            "timestamp_source": "MS2 sensor timestamps; synthetic fallback is not used",
            "image_contract": "thermal_path points to original 16-bit PNG; no copies",
            "detector_config": detector_config,
        },
        output_root / "metadata.json",
    )
    pending = [row for row in rows if _frame_key(row) not in processed]
    with (
        frame_path.open("a", encoding="utf-8") as frame_handle,
        detection_path.open("a", encoding="utf-8") as detection_handle,
        rejection_path.open("a", encoding="utf-8") as rejection_handle,
    ):
        for batch_start in range(0, len(pending), args.batch_size):
            batch_rows = pending[batch_start : batch_start + args.batch_size]
            good_rows: list[dict[str, Any]] = []
            for row in batch_rows:
                if _timestamp_bad(row, args.max_timestamp_delta_ms):
                    frame, accepted, rejected = process_frame(
                        row, [], project_root, args, calibration_cache
                    )
                    _write_jsonl_line(frame_handle, frame)
                    for record in accepted:
                        _write_jsonl_line(detection_handle, record)
                    for record in rejected:
                        _write_jsonl_line(rejection_handle, record)
                else:
                    good_rows.append(row)
            if good_rows:
                images = []
                model_metadata = []
                for row in good_rows:
                    rgb_path = resolve_project_path(row["rgb_path"], project_root)
                    if rgb_path is None or not rgb_path.is_file():
                        raise FileNotFoundError(row["rgb_path"])
                    with Image.open(rgb_path) as image:
                        rgb = np.asarray(image.convert("RGB")).copy()
                    images.append(torch_module.from_numpy(rgb).permute(2, 0, 1).contiguous())
                    model_metadata.append({"height": int(row["rgb_height"]), "width": int(row["rgb_width"])})
                outputs = detector.predict_batch(images, model_metadata)
                for row, model_detections in zip(good_rows, outputs, strict=True):
                    frame, accepted, rejected = process_frame(
                        row, model_detections, project_root, args, calibration_cache
                    )
                    _write_jsonl_line(frame_handle, frame)
                    for record in accepted:
                        _write_jsonl_line(detection_handle, record)
                    for record in rejected:
                        _write_jsonl_line(rejection_handle, record)
            frame_handle.flush()
            detection_handle.flush()
            rejection_handle.flush()
            completed = min(batch_start + len(batch_rows), len(pending))
            if completed % (args.log_every * args.batch_size) < args.batch_size or completed == len(pending):
                print(
                    f"processed {completed}/{len(pending)} pending frames; "
                    f"shard={args.shard}",
                    flush=True,
                )
    print(f"shard complete: {len(pending)} frames")


def _merge_shard_rows(shard_roots: Sequence[Path], name: str) -> Iterable[dict[str, Any]]:
    for shard_root in shard_roots:
        path = shard_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        yield from _read_jsonl(path)


def write_records(
    output_root: Path,
    frame_rows: list[dict[str, Any]],
    detection_rows: list[dict[str, Any]],
) -> None:
    detections_by_key: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detection_rows:
        detections_by_key[_frame_key(detection)].append(detection)
    records: list[dict[str, Any]] = []
    for frame in frame_rows:
        detections = detections_by_key[_frame_key(frame)]
        records.append(
            {
                "id": f"ms2:{frame['sequence']}:{frame['frame_id']}",
                "dataset": "MS2",
                "sequence": frame["sequence"],
                "frame_id": frame["frame_id"],
                "image": frame["thermal_path"],
                "image_path": frame["thermal_path"],
                "width": int(frame["thermal_width"]),
                "height": int(frame["thermal_height"]),
                "image_bits": 16,
                "box_format": "xyxy_abs",
                "boxes": [row["bbox_thr_xyxy"] for row in detections],
                "labels": [row["class_name"] for row in detections],
                "scores": [float(row["score"]) for row in detections],
                "track_ids": [None for _ in detections],
                "track_ids_source": "not_available_for_zero_shot_detections",
                "rgb_path": frame["rgb_path"],
                "timestamp_rgb": frame["timestamp_rgb"],
                "timestamp_thr": frame["timestamp_thr"],
                "timestamp_delta_ms": frame["timestamp_delta_ms"],
                "timestamp_source": "MS2 sensor timestamps",
                "annotation_source": "sam3.1_rgb+ms2_calib+proj_depth_refined",
                "frame_status": frame["status"],
            }
        )
    atomic_jsonl_dump(records, output_root / "records.jsonl")


def write_coco(output_root: Path, frame_rows: list[dict[str, Any]], detection_rows: list[dict[str, Any]]) -> None:
    image_ids = {_frame_key(row): index for index, row in enumerate(frame_rows, start=1)}
    images = [
        {
            "id": image_ids[_frame_key(row)],
            "file_name": row["thermal_path"],
            "width": int(row["thermal_width"]),
            "height": int(row["thermal_height"]),
            "sequence": row["sequence"],
            "frame_id": row["frame_id"],
        }
        for row in frame_rows
    ]
    annotations = []
    for annotation_id, row in enumerate(detection_rows, start=1):
        x1, y1, x2, y2 = (float(value) for value in row["bbox_thr_xyxy"])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_ids[_frame_key(row)],
                "category_id": CATEGORY_IDS[row["class_name"]],
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "score": float(row["score"]),
                "source": row["source"],
                "bbox_rgb_xyxy": row["bbox_rgb_xyxy"],
                "depth_valid_count": row.get("depth_valid_count"),
                "thermal_box_depth_valid_count": row.get("thermal_box_depth_valid_count"),
            }
        )
    atomic_json_dump(
        {
            "images": images,
            "annotations": annotations,
            "categories": [
                {
                    "id": CATEGORY_IDS[name],
                    "name": name,
                    "supercategory": name.split(".", maxsplit=1)[0],
                }
                for name in CATEGORIES
            ],
        },
        output_root / "instances.json",
    )


def write_report(output_root: Path, frame_rows: list[dict[str, Any]], detection_rows: list[dict[str, Any]], rejection_rows: list[dict[str, Any]]) -> None:
    status = Counter(str(row["status"]) for row in frame_rows)
    classes = Counter(str(row["class_name"]) for row in detection_rows)
    reasons = Counter(str(row["reason"]) for row in rejection_rows)
    scores = [float(row["score"]) for row in detection_rows]
    report = {
        "status": "complete",
        "frames_total": len(frame_rows),
        "frames_with_accepted_boxes": sum(row["accepted_count"] > 0 for row in frame_rows),
        "detections_total": len(detection_rows),
        "frame_status": dict(sorted(status.items())),
        "detections_by_class": dict(sorted(classes.items())),
        "rejected_by_reason": dict(sorted(reasons.items())),
        "score": {
            "min": min(scores) if scores else None,
            "median": float(np.median(scores)) if scores else None,
            "p95": float(np.percentile(scores, 95)) if scores else None,
            "max": max(scores) if scores else None,
        },
        "raw_images_are_16bit": True,
        "bbox_coordinate_system": "left thermal pixel xyxy; instances.json uses xywh",
        "manual_review_required": True,
        "full_scale_labeling": True,
    }
    atomic_json_dump(report, output_root / "quality_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def render_previews(output_root: Path, project_root: Path, manifest_path: Path, frame_rows: list[dict[str, Any]], detection_rows: list[dict[str, Any]], preview_per_sequence: int) -> None:
    manifest_rows = list(_read_jsonl(manifest_path))
    by_sequence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest_rows:
        by_sequence[str(row["sequence"])].append(row)
    selected_keys: set[str] = set()
    for sequence, rows in by_sequence.items():
        positions = np.linspace(0, len(rows) - 1, min(preview_per_sequence, len(rows)), dtype=np.int64)
        selected_keys.update(_frame_key(rows[int(position)]) for position in sorted(set(positions)))
    frames_by_key = {_frame_key(row): row for row in frame_rows if _frame_key(row) in selected_keys}
    detections_by_key: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detection_rows:
        key = _frame_key(row)
        if key in selected_keys:
            detections_by_key[key].append(row)
    preview_root = output_root / "visualizations" / "review"
    for row in frames_by_key.values():
        filename = f"{row['sequence']}_{row['frame_id']}.jpg"
        save_visualization(row, detections_by_key[_frame_key(row)], preview_root / filename, project_root)
    print(f"rendered {len(frames_by_key)} representative previews")


def command_merge(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve()
    shard_roots = [Path(value).resolve() for value in args.shard_root]
    manifest_path = output_root / "manifest.jsonl"
    frame_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected = {_frame_key(row) for row in _read_jsonl(manifest_path)}
    for row in _merge_shard_rows(shard_roots, "frames.jsonl"):
        key = _frame_key(row)
        if key in seen:
            raise ValueError(f"duplicate frame result: {key}")
        seen.add(key)
        frame_rows.append(row)
    missing = expected - seen
    extra = seen - expected
    if missing or extra:
        raise RuntimeError(f"shard coverage mismatch: missing={len(missing)} extra={len(extra)}")
    for row in _merge_shard_rows(shard_roots, "detections.jsonl"):
        detection_rows.append(row)
    for row in _merge_shard_rows(shard_roots, "rejections.jsonl"):
        rejection_rows.append(row)
    frame_rows.sort(key=lambda row: (str(row["sequence"]), int(row["rgb_index"])))
    detection_rows.sort(key=lambda row: (str(row["sequence"]), int(row["rgb_index"]), -float(row["score"])))
    atomic_jsonl_dump(frame_rows, output_root / "frames.jsonl")
    atomic_jsonl_dump(detection_rows, output_root / "annotations.jsonl")
    atomic_jsonl_dump(rejection_rows, output_root / "rejections.jsonl")
    write_records(output_root, frame_rows, detection_rows)
    write_coco(output_root, frame_rows, detection_rows)
    write_report(output_root, frame_rows, detection_rows, rejection_rows)
    render_previews(
        output_root,
        project_root,
        manifest_path,
        frame_rows,
        detection_rows,
        args.preview_per_sequence,
    )
    atomic_json_dump(
        {
            "record_file": "records.jsonl",
            "annotation_file": "annotations.jsonl",
            "coco_file": "instances.json",
            "frame_file": "frames.jsonl",
            "raw_image_root": "data/raw/ms2",
            "raw_image_bits": 16,
            "coordinate_system": "left thermal pixel coordinates",
            "source": "SAM3.1 RGB detection + MS2 timestamp pairing + RGB refined depth + calibration projection",
            "shards": [str(value) for value in shard_roots],
            "frames": len(frame_rows),
            "detections": len(detection_rows),
            "manual_review_required": True,
        },
        output_root / "metadata.json",
    )
    print(f"merged {len(frame_rows)} frames and {len(detection_rows)} detections")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--dataset-root", required=True)
    manifest.add_argument("--project-root", required=True)
    manifest.add_argument("--output-root", required=True)
    manifest.add_argument("--overwrite", action="store_true")
    manifest.set_defaults(func=command_manifest)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--project-root", required=True)
    infer.add_argument("--output-root", required=True)
    infer.add_argument("--sam3-eval-root", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--shard", required=True)
    infer.add_argument("--limit", type=int, default=0)
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--batch-size", type=int, default=2)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--input-resolution", type=int, default=1008)
    infer.add_argument("--precision", default="bfloat16", choices=("bfloat16", "float16"))
    infer.add_argument("--confidence-threshold", type=float, default=0.35)
    infer.add_argument("--nms-iou", type=float, default=0.70)
    infer.add_argument("--max-timestamp-delta-ms", type=float, default=100.0)
    infer.add_argument("--min-thermal-depth-support", type=int, default=8)
    infer.add_argument("--min-thermal-height", type=float, default=8.0)
    infer.add_argument("--min-thermal-area", type=float, default=64.0)
    infer.add_argument("--log-every", type=int, default=100)
    infer.set_defaults(func=command_infer)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--project-root", required=True)
    merge.add_argument("--output-root", required=True)
    merge.add_argument("--shard-root", action="append", required=True)
    merge.add_argument("--preview-per-sequence", type=int, default=5)
    merge.set_defaults(func=command_merge)
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
