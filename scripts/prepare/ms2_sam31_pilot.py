#!/usr/bin/env python3
"""Run a small SAM3.1 MS2 RGB-to-thermal bbox pilot.

SAM3.1 detects objects in left RGB frames. MS2 refined RGB depth and the
per-sequence calibration transfer each RGB detection into left thermal pixel
coordinates. Raw frames are read in place and never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

import ms2_pseudo_bbox as geometry


DEFAULT_PROMPTS = ("person", "car", "truck", "bus", "motorcycle", "bicycle")


def chunks(items: Sequence[geometry.Sample], size: int) -> list[Sequence[geometry.Sample]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_detector(args: argparse.Namespace):
    import torch

    sam3_eval_root = Path(args.sam3_eval_root).resolve()
    sys.path.insert(0, str(sam3_eval_root))
    from sam31_detector import Sam31Detector

    checkpoint = Path(args.checkpoint).resolve()
    config = {
        "input_resolution": args.input_resolution,
        "mask_resolution": 256,
        "save_masks": False,
        "prompts": list(args.prompts),
        "confidence_threshold": args.confidence_threshold,
        "max_detections_per_prompt": 100,
        "max_detections_per_image": 150,
        "nms_iou_threshold": args.nms_iou,
        "mask_threshold": 0.5,
        "precision": args.precision,
        "checkpoint_minimum_coverage": 0.95,
        "checkpoint_mmap": True,
    }
    device = torch.device(args.device)
    detector = Sam31Detector(
        config=config,
        device=device,
        sam3_repo=str(sam3_eval_root / "sam3"),
        checkpoint=str(checkpoint),
    )
    return detector, torch, config


def run_pilot(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else geometry.infer_project_root(dataset_root)
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = geometry.build_manifest(
        dataset_root,
        project_root,
        args.frames_per_sequence,
        args.review_frames_per_sequence,
    )
    if args.limit > 0:
        manifest = manifest[: args.limit]
    geometry.atomic_jsonl_dump([asdict(sample) for sample in manifest], output_root / "pilot_manifest.jsonl")

    checkpoint = Path(args.checkpoint).resolve()
    metadata = {
        "backend": "sam3.1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "sam3_eval_root": str(Path(args.sam3_eval_root).resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "input_resolution": args.input_resolution,
        "precision": args.precision,
        "prompts": list(args.prompts),
        "confidence_threshold": args.confidence_threshold,
        "min_thermal_height": args.min_thermal_height,
        "min_thermal_area": args.min_thermal_area,
        "nms_iou": args.nms_iou,
        "frames": len(manifest),
        "sequences": sorted({sample.sequence for sample in manifest}),
        "coordinate_contract": "bbox_thr_xyxy is in left thermal image pixels; bbox_rgb_xyxy is in left RGB image pixels",
        "geometry_contract": "RGB refined depth in meters; MS2 calibration translation in millimeters",
        "manual_review_required": True,
        "full_scale_started": False,
    }
    geometry.atomic_json_dump(metadata, output_root / "metadata.json")

    detector, torch, detector_config = make_detector(args)
    calibration_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    frame_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    visualization_root = output_root / "visualizations" / "review"

    for batch_number, batch in enumerate(chunks(manifest, args.batch_size), start=1):
        images: list[torch.Tensor] = []
        metadata_rows: list[dict[str, Any]] = []
        sample_rows: list[dict[str, Any]] = []
        runtime_rows: list[dict[str, Any]] = []
        for sample in batch:
            sample_dict = asdict(sample)
            rgb_path = geometry.resolve_project_path(sample.rgb_path, project_root)
            thermal_path = geometry.resolve_project_path(sample.thermal_path, project_root)
            depth_path = geometry.resolve_project_path(sample.depth_rgb_path, project_root)
            depth_thr_path = geometry.resolve_project_path(sample.depth_thr_path, project_root)
            calib_path = geometry.resolve_project_path(sample.calib_path, project_root)
            if not rgb_path or not thermal_path or not calib_path:
                raise RuntimeError(f"incomplete manifest row: {sample}")
            rgb_image = Image.open(rgb_path).convert("RGB")
            thermal_image = Image.open(thermal_path)
            rgb_array = np.asarray(rgb_image)
            rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1).contiguous()
            images.append(rgb_tensor)
            sample_dict.update(
                {
                    "rgb_width": rgb_image.width,
                    "rgb_height": rgb_image.height,
                    "thermal_width": thermal_image.width,
                    "thermal_height": thermal_image.height,
                }
            )
            sample_rows.append(sample_dict)
            metadata_rows.append(
                {
                    "height": rgb_image.height,
                    "width": rgb_image.width,
                }
            )
            runtime_rows.append(
                {
                    "rgb_path": rgb_path,
                    "thermal_path": thermal_path,
                    "depth_path": depth_path,
                    "depth_thr_path": depth_thr_path,
                    "calib_path": calib_path,
                    "rgb_size": rgb_image.size,
                    "thermal_size": thermal_image.size,
                }
            )

        outputs = detector.predict_batch(images, metadata_rows)
        for sample_dict, runtime, detections in zip(sample_rows, runtime_rows, outputs):
            calib_path = runtime["calib_path"]
            if calib_path not in calibration_cache:
                calibration_cache[calib_path] = geometry.load_calibration(calib_path)
            k_rgb, k_thr, rgb_to_thr = calibration_cache[calib_path]
            depth_rgb = None
            depth_scale = 1.0
            if runtime["depth_path"] is not None and runtime["depth_path"].is_file():
                depth_rgb, depth_scale = geometry.load_depth(runtime["depth_path"])
            else:
                rejection_rows.append({**sample_dict, "reason": "missing_rgb_depth"})

            depth_thr = None
            thermal_depth_scale = 1.0
            if runtime["depth_thr_path"] is not None and runtime["depth_thr_path"].is_file():
                depth_thr, thermal_depth_scale = geometry.load_depth(runtime["depth_thr_path"])

            accepted: list[dict[str, Any]] = []
            candidates = []
            for detection in detections:
                class_name = geometry.canonical_class(str(detection["prompt"]))
                if class_name is None:
                    rejection_rows.append(
                        {
                            **sample_dict,
                            "prompt": str(detection["prompt"]),
                            "score": float(detection["score"]),
                            "reason": "unsupported_prompt",
                        }
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
            candidates = geometry.classwise_nms(candidates, args.nms_iou)
            for candidate in candidates:
                record = {**sample_dict, **candidate}
                try:
                    if depth_rgb is None:
                        raise geometry.GeometryFailure("missing_rgb_depth")
                    projected = geometry.transfer_rgb_box_to_thermal(
                        candidate["bbox_rgb_xyxy"],
                        depth_rgb,
                        depth_scale,
                        k_rgb,
                        k_thr,
                        rgb_to_thr,
                        runtime["rgb_size"],
                        runtime["thermal_size"],
                    )
                    record.update(projected)
                    tx1, ty1, tx2, ty2 = projected["bbox_thr_xyxy"]
                    thermal_box_width = float(tx2 - tx1)
                    thermal_box_height = float(ty2 - ty1)
                    thermal_box_area = thermal_box_width * thermal_box_height
                    if (
                        thermal_box_height < args.min_thermal_height
                        or thermal_box_area < args.min_thermal_area
                    ):
                        raise geometry.GeometryFailure(
                            "thermal_box_below_quality_gate",
                            {
                                "thermal_box_width": thermal_box_width,
                                "thermal_box_height": thermal_box_height,
                                "thermal_box_area": thermal_box_area,
                            },
                        )
                    record["bbox_thr_xyxy"] = projected.pop("bbox_thr_xyxy")
                    record["source"] = "sam3.1_rgb+ms2_calib+proj_depth_refined"
                    if depth_thr is not None:
                        tx1, ty1, tx2, ty2 = record["bbox_thr_xyxy"]
                        x0 = max(0, int(np.floor(tx1)))
                        x1 = min(depth_thr.shape[1], int(np.ceil(tx2)))
                        y0 = max(0, int(np.floor(ty1)))
                        y1 = min(depth_thr.shape[0], int(np.ceil(ty2)))
                        if x1 > x0 and y1 > y0:
                            thermal_values = depth_thr[y0:y1, x0:x1] * thermal_depth_scale
                            thermal_valid = thermal_values[
                                np.isfinite(thermal_values) & (thermal_values > 0.1)
                            ]
                            record["thermal_box_depth_valid_count"] = int(thermal_valid.size)
                            record["thermal_box_depth_median_m"] = (
                                float(np.median(thermal_valid)) if thermal_valid.size else None
                            )
                    accepted.append(record)
                except geometry.GeometryFailure as failure:
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
            if bool(sample_dict["review"]):
                filename = f"{len(frame_rows):05d}_{sample_dict['sequence']}_{sample_dict['frame_id']}.jpg"
                geometry.save_visualization(sample_dict, accepted, visualization_root / filename, project_root)
        print(
            f"processed batch {batch_number}/{math.ceil(len(manifest) / args.batch_size)}; "
            f"frames={len(frame_rows)} accepted_boxes={len(detection_rows)}",
            flush=True,
        )

    geometry.atomic_jsonl_dump(frame_rows, output_root / "pilot_frames.jsonl")
    geometry.atomic_jsonl_dump(detection_rows, output_root / "pilot_detections.jsonl")
    geometry.atomic_jsonl_dump(rejection_rows, output_root / "pilot_rejections.jsonl")
    geometry.write_coco_dataset(output_root)
    geometry.write_quality_report(output_root)
    geometry.atomic_json_dump({**metadata, "detector_config": detector_config}, output_root / "metadata.json")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--project-root")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sam3-eval-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=20)
    parser.add_argument("--review-frames-per-sequence", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-resolution", type=int, default=1008)
    parser.add_argument("--precision", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--min-thermal-height", type=float, default=8.0)
    parser.add_argument("--min-thermal-area", type=float, default=64.0)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    args.prompts = tuple(args.prompts or DEFAULT_PROMPTS)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        run_pilot(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
