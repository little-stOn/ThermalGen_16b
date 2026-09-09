#!/usr/bin/env python3
"""Create a non-destructive high-confidence MS2 bbox dataset view."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from ms2_pseudo_bbox import atomic_json_dump, atomic_jsonl_dump, save_visualization


CATEGORY_NAMES = (
    "human.pedestrian",
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus",
    "vehicle.motorcycle",
    "vehicle.bicycle",
)
CATEGORY_IDS = {name: index for index, name in enumerate(CATEGORY_NAMES, start=1)}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def frame_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["sequence"]), str(row["frame_id"])


def filtered_record(
    frame: dict[str, Any], detections: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    return {
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
        "confidence_filter": {"operator": ">", "threshold": threshold},
        "frame_status": frame["status"],
    }


def write_coco(
    output_root: Path,
    frame_rows: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> None:
    image_ids = {frame_key(row): index for index, row in enumerate(frame_rows, start=1)}
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
    annotations = []
    for annotation_id, row in enumerate(detections, start=1):
        x1, y1, x2, y2 = (float(value) for value in row["bbox_thr_xyxy"])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_ids[frame_key(row)],
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
                for name in CATEGORY_NAMES
            ],
        },
        output_root / "instances.json",
    )


def render_previews(
    output_root: Path,
    project_root: Path,
    frame_rows: list[dict[str, Any]],
    detections_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    preview_per_sequence: int,
) -> int:
    by_sequence: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_sequence[str(row["sequence"])].append(row)
    preview_root = output_root / "visualizations" / "review"
    selected: list[dict[str, Any]] = []
    for rows in by_sequence.values():
        positions = np.linspace(0, len(rows) - 1, min(preview_per_sequence, len(rows)), dtype=np.int64)
        selected.extend(rows[int(position)] for position in sorted(set(positions)))
    for row in selected:
        save_visualization(
            row,
            detections_by_key.get(frame_key(row), []),
            preview_root / f"{row['sequence']}_{row['frame_id']}.jpg",
            project_root,
        )
    return len(selected)


def run(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    project_root = Path(args.project_root).resolve()
    threshold = float(args.threshold)
    if not output_root.exists():
        output_root.mkdir(parents=True)
    elif any(output_root.iterdir()):
        raise FileExistsError(f"output is not empty: {output_root}")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")

    frame_rows = list(read_jsonl(input_root / "frames.jsonl"))
    high_by_key: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    filtered_detections: list[dict[str, Any]] = []
    rejected_detections: list[dict[str, Any]] = []
    total_detections = 0
    for row in read_jsonl(input_root / "annotations.jsonl"):
        total_detections += 1
        if float(row["score"]) > threshold:
            filtered_detections.append(row)
            high_by_key[frame_key(row)].append(row)
        else:
            rejected_detections.append({**row, "reason": "confidence_below_threshold", "threshold": threshold})

    def updated_frames() -> Iterable[dict[str, Any]]:
        for row in frame_rows:
            count = len(high_by_key.get(frame_key(row), []))
            updated = {
                **row,
                "pre_filter_accepted_count": row.get("accepted_count", 0),
                "accepted_count": count,
                "confidence_filter": {"operator": ">", "threshold": threshold},
            }
            if count:
                updated["status"] = "accepted_highconf"
            elif int(row.get("accepted_count", 0)):
                updated["status"] = "confidence_filtered_empty"
            yield updated

    updated_frame_rows = list(updated_frames())
    write_jsonl(updated_frame_rows, output_root / "frames.jsonl")
    write_jsonl(filtered_detections, output_root / "annotations.jsonl")
    write_jsonl(
        list(read_jsonl(input_root / "rejections.jsonl")) + rejected_detections,
        output_root / "rejections.jsonl",
    )
    write_jsonl(
        (
            filtered_record(frame, high_by_key.get(frame_key(frame), []), threshold)
            for frame in updated_frame_rows
        ),
        output_root / "records.jsonl",
    )
    write_coco(output_root, updated_frame_rows, filtered_detections)
    detections_by_key = {key: rows for key, rows in high_by_key.items()}
    preview_count = render_previews(
        output_root,
        project_root,
        updated_frame_rows,
        detections_by_key,
        args.preview_per_sequence,
    )

    source_metadata = json.loads((input_root / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "parent_output": str(input_root),
        "record_file": "records.jsonl",
        "annotation_file": "annotations.jsonl",
        "coco_file": "instances.json",
        "frame_file": "frames.jsonl",
        "raw_image_root": "data/raw/ms2",
        "raw_image_bits": 16,
        "score_filter": {"operator": ">", "threshold": threshold},
        "source_detections": total_detections,
        "filtered_detections": len(filtered_detections),
        "raw_images_copied": False,
        "manual_review_required": True,
    }
    atomic_json_dump(metadata, output_root / "metadata.json")
    class_counts = Counter(row["class_name"] for row in filtered_detections)
    status_counts = Counter(row["status"] for row in updated_frame_rows)
    report = {
        "status": "complete",
        "score_filter": {"operator": ">", "threshold": threshold},
        "frames_total": len(updated_frame_rows),
        "frames_with_highconf_boxes": sum(row["accepted_count"] > 0 for row in updated_frame_rows),
        "detections_total": len(filtered_detections),
        "source_detections": total_detections,
        "retention": len(filtered_detections) / total_detections if total_detections else 0.0,
        "detections_by_class": dict(sorted(class_counts.items())),
        "frame_status": dict(sorted(status_counts.items())),
        "rejected_low_confidence": len(rejected_detections),
        "review_visualizations": preview_count,
        "raw_images_are_16bit": True,
        "raw_images_copied": False,
        "bbox_coordinate_system": "left thermal pixel xyxy; instances.json uses xywh",
        "manual_review_required": True,
    }
    atomic_json_dump(report, output_root / "quality_report.json")
    shutil.copyfile(input_root / "manifest.jsonl", output_root / "manifest.jsonl")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--preview-per-sequence", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
