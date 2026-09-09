#!/usr/bin/env python3
"""Merge sharded ViViD++ thermal annotation artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from typing import Any, Sequence


CATEGORY_NAMES = (
    "human.pedestrian",
    "vehicle.bicycle",
    "vehicle.bus",
    "vehicle.car",
    "vehicle.motorcycle",
    "vehicle.truck",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _copy_frame_assets(
    shard_root: Path,
    output_root: Path,
    shard_name: str,
    frame_index: int,
    frame: dict[str, Any],
) -> None:
    for modality in ("rgb", "thermal"):
        source = shard_root / frame[f"{modality}_path"]
        destination = output_root / "images" / shard_name / modality / f"{frame_index:08d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        frame[f"{modality}_path"] = str(destination.relative_to(output_root))
    for modality in ("rgb", "thermal"):
        source = shard_root / "previews" / modality / f"{frame.get('_shard_frame_index', 0):06d}.jpg"
        if source.is_file():
            destination = output_root / "previews" / modality / f"{frame_index:08d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def _write_coco(
    output_root: Path,
    frames: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> None:
    category_ids = {name: index for index, name in enumerate(CATEGORY_NAMES, start=1)}
    images = []
    image_ids: dict[tuple[str, float], int] = {}
    for index, frame in enumerate(frames, start=1):
        image_ids[(str(frame["bag"]), float(frame["thermal_stamp"]))] = index
        images.append(
            {
                "id": index,
                "file_name": frame["thermal_path"],
                "width": frame["thermal_width"],
                "height": frame["thermal_height"],
            }
        )
    annotations = []
    for index, detection in enumerate(detections, start=1):
        x1, y1, x2, y2 = detection["bbox_thr_xyxy"]
        annotations.append(
            {
                "id": index,
                "image_id": image_ids[(str(detection["bag"]), float(detection["thermal_stamp"]))],
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


def merge(input_roots: Sequence[Path], output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output exists and is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    merged_frames: list[dict[str, Any]] = []
    merged_detections: list[dict[str, Any]] = []
    merged_rejections: list[dict[str, Any]] = []
    shard_metadata: list[dict[str, Any]] = []
    for shard_number, shard_root in enumerate(input_roots):
        shard_root = shard_root.resolve()
        shard_name = f"shard{shard_number:02d}"
        frames = _read_jsonl(shard_root / "frames.jsonl")
        detections = _read_jsonl(shard_root / "detections.jsonl")
        rejections = _read_jsonl(shard_root / "rejections.jsonl")
        by_key: dict[tuple[str, float], dict[str, Any]] = {}
        for index, frame in enumerate(frames):
            frame["_shard_frame_index"] = index
            global_index = len(merged_frames)
            _copy_frame_assets(shard_root, output_root, shard_name, global_index, frame)
            by_key[(str(frame["bag"]), float(frame["thermal_stamp"]))] = frame
            merged_frames.append(frame)
        for detection in detections:
            frame = by_key[(str(detection["bag"]), float(detection["thermal_stamp"]))]
            detection["rgb_path"] = frame["rgb_path"]
            detection["thermal_path"] = frame["thermal_path"]
            merged_detections.append(detection)
        merged_rejections.extend(rejections)
        metadata_path = shard_root / "metadata.json"
        if metadata_path.is_file():
            shard_metadata.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    for frame in merged_frames:
        frame.pop("_shard_frame_index", None)
    _write_jsonl(output_root / "frames.jsonl", merged_frames)
    _write_jsonl(output_root / "detections.jsonl", merged_detections)
    _write_jsonl(output_root / "rejections.jsonl", merged_rejections)
    _write_coco(output_root, merged_frames, merged_detections)
    first = shard_metadata[0] if shard_metadata else {}
    metadata = {
        "backend": "sam3.1",
        "target_modality": "thermal",
        "rgb_role": "transition_only",
        "shards": [str(path.resolve()) for path in input_roots],
        "shard_metadata": shard_metadata,
        "frames": len(merged_frames),
        "detections": len(merged_detections),
        "rejections": len(merged_rejections),
        "class_counts": dict(Counter(row["class_name"] for row in merged_detections)),
        "detector_config": first.get("detector_config"),
        "thermal_margin_ratio": first.get("thermal_margin_ratio"),
        "bbox_transfer_contract": first.get("bbox_transfer_contract"),
        "manual_review_required": True,
        "production_rollout_allowed": False,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    merge(args.input_root, args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
