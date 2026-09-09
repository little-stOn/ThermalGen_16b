#!/usr/bin/env python3
"""Apply conservative, model-free QC to an MS2 SAM3.1 pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import ms2_pseudo_bbox as geometry


def read_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_qc(
    input_root: Path,
    output_root: Path,
    project_root: Path,
    max_timestamp_delta_ms: float,
    min_thermal_depth_support: int,
) -> None:
    frames = geometry.read_jsonl(input_root / "pilot_frames.jsonl")
    detections = geometry.read_jsonl(input_root / "pilot_detections.jsonl")
    rejections = geometry.read_jsonl(input_root / "pilot_rejections.jsonl")
    frame_key = lambda row: (str(row["sequence"]), str(row["frame_id"]))
    kept_frame_rows: list[dict[str, Any]] = []
    excluded_frames: list[dict[str, Any]] = []
    kept_keys: set[tuple[str, str]] = set()
    for row in frames:
        delta = row.get("timestamp_delta_ms")
        timestamp_ok = delta is None or abs(float(delta)) <= max_timestamp_delta_ms
        if timestamp_ok:
            kept_keys.add(frame_key(row))
            kept_frame_rows.append(dict(row))
        else:
            excluded_frames.append({**row, "reason": "timestamp_delta_exceeds_qc"})

    kept_by_key: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    qc_rejections = list(rejections)
    for row in detections:
        key = frame_key(row)
        thermal_support = int(row.get("thermal_box_depth_valid_count", 0))
        if key not in kept_keys:
            qc_rejections.append({**row, "reason": "timestamp_delta_exceeds_qc"})
        elif thermal_support < min_thermal_depth_support:
            qc_rejections.append({**row, "reason": "thermal_depth_support_below_qc"})
        else:
            kept_by_key[key].append(row)

    for row in kept_frame_rows:
        accepted = kept_by_key[frame_key(row)]
        row["qc_accepted_count"] = len(accepted)
        row["accepted_count"] = len(accepted)
        row["status"] = "qc_accepted" if accepted else "qc_empty"

    geometry.atomic_jsonl_dump(
        [row for row in geometry.read_jsonl(input_root / "pilot_manifest.jsonl") if frame_key(row) in kept_keys],
        output_root / "pilot_manifest.jsonl",
    )
    geometry.atomic_jsonl_dump(kept_frame_rows, output_root / "pilot_frames.jsonl")
    geometry.atomic_jsonl_dump(
        [row for values in kept_by_key.values() for row in values],
        output_root / "pilot_detections.jsonl",
    )
    geometry.atomic_jsonl_dump(qc_rejections + excluded_frames, output_root / "pilot_rejections.jsonl")
    geometry.write_coco_dataset(output_root)

    visualization_root = output_root / "visualizations" / "review"
    for ordinal, row in enumerate(kept_frame_rows, start=1):
        if row.get("review"):
            geometry.save_visualization(
                row,
                kept_by_key[frame_key(row)],
                visualization_root / f"{ordinal:05d}_{row['sequence']}_{row['frame_id']}.jpg",
                project_root,
            )

    source_metadata = read_metadata(input_root / "metadata.json")
    qc_info = {
        "source_root": str(input_root),
        "max_timestamp_delta_ms": max_timestamp_delta_ms,
        "min_thermal_depth_support": min_thermal_depth_support,
        "source_frames": len(frames),
        "kept_frames": len(kept_frame_rows),
        "excluded_frames": len(excluded_frames),
        "source_detections": len(detections),
        "kept_detections": sum(len(values) for values in kept_by_key.values()),
        "manual_review_required": True,
        "full_scale_started": False,
    }
    geometry.atomic_json_dump(
        {**source_metadata, "quality_gate": qc_info},
        output_root / "metadata.json",
    )
    geometry.atomic_json_dump(qc_info, output_root / "qc_summary.json")
    geometry.write_quality_report(output_root)
    print(
        json.dumps(
            {
                "source_frames": len(frames),
                "kept_frames": len(kept_frame_rows),
                "excluded_frames": len(excluded_frames),
                "source_detections": len(detections),
                "kept_detections": sum(len(values) for values in kept_by_key.values()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--max-timestamp-delta-ms", type=float, default=100.0)
    parser.add_argument("--min-thermal-depth-support", type=int, default=8)
    args = parser.parse_args(argv)
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"output exists: {output_root}")
    output_root.mkdir(parents=True)
    run_qc(
        input_root,
        output_root,
        Path(args.project_root).resolve(),
        args.max_timestamp_delta_ms,
        args.min_thermal_depth_support,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
