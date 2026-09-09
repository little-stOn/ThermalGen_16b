#!/usr/bin/env python3
"""Evaluate SAM3.1 on a balanced BU-TIV XML ground-truth subset.

BU-TIV contains thermal-only 16-bit images, while SAM3.1 is used here with an
RGB-pretrained checkpoint. Each thermal frame is therefore converted to a
three-channel uint8 view using one global percentile range computed from the
gold subset. The report measures this domain-transfer experiment, not MS2
annotation precision directly.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw


PROMPTS = ("person", "bicycle", "motorcycle", "car")
SCORE_THRESHOLDS = (0.25, 0.35, 0.50)
IOU_THRESHOLDS = (0.50, 0.75)


def import_butiv_common():
    project_root = Path(__file__).resolve().parents[2]
    source_root = project_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from dwm.datasets.butiv_common import load_sequence_records, sequence_names

    return load_sequence_records, sequence_names


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    temporary.replace(path)


def atomic_jsonl_dump(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=_json_default))
            handle.write("\n")
    temporary.replace(path)


def load_uint16(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image.load()
        if image.mode not in {"I;16", "I;16B", "I"}:
            raise ValueError(f"BU-TIV image is not 16-bit grayscale: {path} ({image.mode})")
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"BU-TIV image is not single-channel: {path} {array.shape}")
    return array.astype(np.uint16, copy=False)


def select_gold_frames(dataset_root: Path, frames_per_sequence: int) -> list[Any]:
    load_sequence_records, sequence_names = import_butiv_common()
    if frames_per_sequence < 1:
        raise ValueError("frames_per_sequence must be positive")
    frames: list[Any] = []
    for sequence in sequence_names("all"):
        records = load_sequence_records(dataset_root, sequence)
        positions = np.linspace(0, len(records) - 1, frames_per_sequence, dtype=np.int64)
        for position in sorted(set(int(value) for value in positions)):
            frames.append(records[position])
    return frames


def global_uint16_range(records: Sequence[Any], low_percentile: float, high_percentile: float) -> tuple[float, float]:
    samples: list[np.ndarray] = []
    for record in records:
        array = load_uint16(record.image_path).reshape(-1)
        stride = max(1, array.size // 20_000)
        samples.append(array[::stride])
    values = np.concatenate(samples).astype(np.float32, copy=False)
    low, high = np.percentile(values, [low_percentile, high_percentile])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        raise ValueError("gold subset has no usable intensity range")
    return float(low), float(high)


def thermal_to_model_tensor(array: np.ndarray, value_range: tuple[float, float], torch_module: Any):
    low, high = value_range
    normalized = np.clip((array.astype(np.float32) - low) * 255.0 / (high - low), 0.0, 255.0)
    rgb = np.repeat(normalized.astype(np.uint8)[..., None], 3, axis=2)
    return torch_module.from_numpy(rgb).permute(2, 0, 1).contiguous()


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

    config = {
        "input_resolution": args.input_resolution,
        "mask_resolution": 256,
        "save_masks": False,
        "prompts": list(PROMPTS),
        "confidence_threshold": args.model_min_score,
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
        checkpoint=str(Path(args.checkpoint).resolve()),
    )
    return detector, torch, config


def canonical_label(value: str) -> str | None:
    normalized = value.strip().lower()
    if "person" in normalized or "pedestrian" in normalized:
        return "human.pedestrian"
    if "bicycle" in normalized or normalized == "bike":
        return "vehicle.bicycle"
    if "motorcycle" in normalized or "motorbike" in normalized:
        return "vehicle.motorcycle"
    if "car" in normalized or "automobile" in normalized:
        return "vehicle.car"
    return None


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


def average_precision(detections: list[dict[str, Any]], ground_truth: dict[str, list[Sequence[float]]], iou_threshold: float) -> float:
    ordered = sorted(detections, key=lambda item: float(item["score"]), reverse=True)
    matched: defaultdict[str, set[int]] = defaultdict(set)
    true_positive: list[int] = []
    false_positive: list[int] = []
    for detection in ordered:
        image_id = str(detection["image_id"])
        gt_boxes = ground_truth[image_id]
        best_index = -1
        best_iou = 0.0
        for index, gt_box in enumerate(gt_boxes):
            if index in matched[image_id]:
                continue
            overlap = bbox_iou(detection["bbox_xyxy"], gt_box)
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_iou >= iou_threshold and best_index >= 0:
            matched[image_id].add(best_index)
            true_positive.append(1)
            false_positive.append(0)
        else:
            true_positive.append(0)
            false_positive.append(1)
    total_gt = sum(len(boxes) for boxes in ground_truth.values())
    if total_gt == 0:
        return 0.0
    if not ordered:
        return 0.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recalls = cumulative_tp / total_gt
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    recall_grid = np.linspace(0.0, 1.0, 101)
    return float(np.mean([np.max(precisions[recalls >= recall]) if np.any(recalls >= recall) else 0.0 for recall in recall_grid]))


def evaluate_at_threshold(
    frames: Sequence[dict[str, Any]],
    predictions: Sequence[dict[str, Any]],
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    classes = sorted({str(row["class_name"]) for row in predictions} | {str(label) for frame in frames for label in frame["gt_labels"]})
    predictions_by_class: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    ground_truth_by_class: defaultdict[str, dict[str, list[Sequence[float]]]] = defaultdict(dict)
    aggregate = Counter()
    per_class: dict[str, Any] = {}
    for frame in frames:
        image_id = str(frame["image_id"])
        for class_name in classes:
            ground_truth_by_class[class_name][image_id] = []
        for box, class_name in zip(frame["gt_boxes"], frame["gt_labels"], strict=True):
            ground_truth_by_class[class_name][image_id].append(box)
    for row in predictions:
        if float(row["score"]) >= score_threshold:
            predictions_by_class[str(row["class_name"])].append(row)

    for class_name in classes:
        matched: defaultdict[str, set[int]] = defaultdict(set)
        tp = fp = 0
        for row in sorted(predictions_by_class[class_name], key=lambda item: float(item["score"]), reverse=True):
            image_id = str(row["image_id"])
            gt_boxes = ground_truth_by_class[class_name][image_id]
            best_index = -1
            best_iou = 0.0
            for index, gt_box in enumerate(gt_boxes):
                if index in matched[image_id]:
                    continue
                overlap = bbox_iou(row["bbox_xyxy"], gt_box)
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_iou >= iou_threshold and best_index >= 0:
                matched[image_id].add(best_index)
                tp += 1
            else:
                fp += 1
        total_gt = sum(len(boxes) for boxes in ground_truth_by_class[class_name].values())
        fn = total_gt - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap = average_precision(
            predictions_by_class[class_name],
            ground_truth_by_class[class_name],
            iou_threshold,
        )
        per_class[class_name] = {
            "gt": total_gt,
            "pred": len(predictions_by_class[class_name]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
        }
        aggregate.update({"gt": total_gt, "pred": len(predictions_by_class[class_name]), "tp": tp, "fp": fp, "fn": fn})
    aggregate_precision = aggregate["tp"] / (aggregate["tp"] + aggregate["fp"]) if aggregate["tp"] + aggregate["fp"] else 0.0
    aggregate_recall = aggregate["tp"] / (aggregate["tp"] + aggregate["fn"]) if aggregate["tp"] + aggregate["fn"] else 0.0
    return {
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "gt": aggregate["gt"],
        "pred": aggregate["pred"],
        "tp": aggregate["tp"],
        "fp": aggregate["fp"],
        "fn": aggregate["fn"],
        "precision": aggregate_precision,
        "recall": aggregate_recall,
        "f1": 2.0 * aggregate_precision * aggregate_recall / (aggregate_precision + aggregate_recall)
        if aggregate_precision + aggregate_recall
        else 0.0,
        "macro_ap": float(np.mean([value["ap"] for value in per_class.values()])) if per_class else 0.0,
        "per_class": per_class,
    }


def thermal_display(array: np.ndarray, value_range: tuple[float, float]) -> Image.Image:
    low, high = value_range
    scaled = np.clip((array.astype(np.float32) - low) * 255.0 / (high - low), 0.0, 255.0)
    return Image.fromarray(scaled.astype(np.uint8), mode="L").convert("RGB")


def draw_preview(frame: dict[str, Any], predictions: list[dict[str, Any]], value_range: tuple[float, float], output_path: Path) -> None:
    image = thermal_display(load_uint16(Path(frame["image_path"])), value_range)
    draw = ImageDraw.Draw(image)
    for box, label in zip(frame["gt_boxes"], frame["gt_labels"], strict=True):
        draw.rectangle(tuple(box), outline=(0, 255, 0), width=3)
        draw.text((box[0] + 2, max(0.0, box[1] - 14.0)), f"GT {label}", fill=(0, 255, 0))
    for row in predictions:
        box = row["bbox_xyxy"]
        draw.rectangle(tuple(box), outline=(255, 60, 0), width=2)
        draw.text((box[0] + 2, box[1] + 2), f"P {row['class_name']} {float(row['score']):.2f}", fill=(255, 60, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=92)


def run(args: argparse.Namespace) -> None:
    import torch

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    records = select_gold_frames(dataset_root, args.frames_per_sequence)
    intensity_range = global_uint16_range(records, args.low_percentile, args.high_percentile)
    detector, torch_module, detector_config = make_detector(args)
    predictions: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    prediction_by_image: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for ordinal, record in enumerate(records, start=1):
        array = load_uint16(record.image_path)
        height, width = array.shape
        image_id = f"{record.sequence}:{record.ordinal:06d}"
        image_tensor = thermal_to_model_tensor(array, intensity_range, torch_module)
        outputs = detector.predict_batch(
            [image_tensor],
            [{"height": height, "width": width}],
        )[0]
        gt_boxes = [list(box) for box in record.boxes]
        gt_labels = [str(label) for label in record.labels]
        frame = {
            "image_id": image_id,
            "sequence": record.sequence,
            "ordinal": record.ordinal,
            "frame_number": record.frame_number,
            "image_path": str(record.image_path),
            "width": width,
            "height": height,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
        }
        frames.append(frame)
        for detection in outputs:
            class_name = canonical_label(str(detection["prompt"]))
            if class_name is None:
                continue
            row = {
                "image_id": image_id,
                "sequence": record.sequence,
                "ordinal": record.ordinal,
                "frame_number": record.frame_number,
                "class_name": class_name,
                "prompt": str(detection["prompt"]),
                "score": float(detection["score"]),
                "bbox_xyxy": [float(value) for value in detection["bbox_xyxy"]],
            }
            predictions.append(row)
            prediction_by_image[image_id].append(row)
        if ordinal % 10 == 0 or ordinal == len(records):
            print(f"processed {ordinal}/{len(records)} frames; predictions={len(predictions)}", flush=True)

    atomic_jsonl_dump(
        [
            {
                "image_id": frame["image_id"],
                "sequence": frame["sequence"],
                "ordinal": frame["ordinal"],
                "frame_number": frame["frame_number"],
                "image_path": frame["image_path"],
                "width": frame["width"],
                "height": frame["height"],
                "gt_boxes": frame["gt_boxes"],
                "gt_labels": frame["gt_labels"],
            }
            for frame in frames
        ],
        output_root / "gold_frames.jsonl",
    )
    atomic_jsonl_dump(predictions, output_root / "gold_predictions.jsonl")
    report_metrics = [
        evaluate_at_threshold(frames, predictions, score_threshold, iou_threshold)
        for score_threshold in SCORE_THRESHOLDS
        for iou_threshold in IOU_THRESHOLDS
    ]
    report = {
        "status": "complete",
        "evaluation": "SAM3.1 RGB-pretrained checkpoint on normalized BU-TIV thermal frames",
        "frames": len(frames),
        "sequences": sorted({frame["sequence"] for frame in frames}),
        "gt_boxes": sum(len(frame["gt_boxes"]) for frame in frames),
        "raw_predictions": len(predictions),
        "model_min_score": args.model_min_score,
        "intensity_range": list(intensity_range),
        "intensity_range_percentiles": [args.low_percentile, args.high_percentile],
        "metrics": report_metrics,
        "limitations": [
            "This evaluates direct thermal-to-RGB-domain transfer, not the MS2 RGB-plus-calibration pipeline.",
            "BU-TIV XML boxes are used as gold labels; no manual relabeling was added.",
            "AP is computed on this balanced subset and is not a full-dataset estimate.",
        ],
    }
    atomic_json_dump(report, output_root / "gold_report.json")
    atomic_json_dump(
        {
            "backend": "sam3.1",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": checkpoint_sha256(Path(args.checkpoint).resolve()),
            "sam3_eval_root": str(Path(args.sam3_eval_root).resolve()),
            "prompts": list(PROMPTS),
            "detector_config": detector_config,
            "frames_per_sequence": args.frames_per_sequence,
            "intensity_range": list(intensity_range),
            "intensity_range_source": "global gold-subset percentiles",
            "timestamp_source": "BU-TIV synthetic ordinal; evaluation is frame-aligned by XML/image order",
        },
        output_root / "metadata.json",
    )
    for frame in frames:
        if frame["ordinal"] % max(1, len(frames) // max(1, args.preview_count)) == 1:
            draw_preview(
                frame,
                prediction_by_image[frame["image_id"]],
                intensity_range,
                output_root / "visualizations" / f"{frame['sequence']}_{frame['ordinal']:06d}.jpg",
            )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--sam3-eval-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-per-sequence", type=int, default=20)
    parser.add_argument("--preview-count", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-resolution", type=int, default=1008)
    parser.add_argument("--precision", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--model-min-score", type=float, default=0.15)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--low-percentile", type=float, default=1.0)
    parser.add_argument("--high-percentile", type=float, default=99.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
