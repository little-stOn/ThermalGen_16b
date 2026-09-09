#!/usr/bin/env python3
"""Visualize DataLoader batches and verify temporal sample coherence."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - config parsing without PyTorch
    torch = None  # type: ignore[assignment]

from dwm.common import load_dataloader_from_config, load_task_dataloader


_FRAME_NUMBER_RE = re.compile(r"(?:frame(?:index)?[-_])(\d+)", re.IGNORECASE)
_TRAILING_NUMBER_RE = re.compile(r"(\d+)$")


def _shape(value: Any) -> tuple[int, ...]:
    if torch is not None and torch.is_tensor(value):
        return tuple(int(size) for size in value.shape)
    dimensions: list[int] = []
    while isinstance(value, (list, tuple)):
        dimensions.append(len(value))
        if not value:
            break
        value = value[0]
    return tuple(dimensions)


def _at(value: Any, *indices: int) -> Any:
    if torch is not None and torch.is_tensor(value):
        return value[indices]
    for index in indices:
        value = value[index]
    return value


def _scalar(value: Any) -> float:
    if torch is not None and torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _bool_at(value: Any, *indices: int) -> bool:
    return bool(_scalar(_at(value, *indices)))


def _frame_key(frame_id: str) -> tuple[int, int | str]:
    text = str(frame_id)
    match = _FRAME_NUMBER_RE.search(text) or _TRAILING_NUMBER_RE.search(text)
    if match is not None:
        return 0, int(match.group(1))
    return 1, text


def parse_sample_id(sample_id: str) -> dict[str, str]:
    parts = str(sample_id).split(":")
    if len(parts) < 4:
        raise ValueError(
            "sample_id must contain dataset, sequence, frame, and view: "
            f"{sample_id!r}"
        )
    return {
        "raw": str(sample_id),
        "dataset": parts[0],
        "sequence": ":".join(parts[1:-2]),
        "frame_id": parts[-2],
        "view": parts[-1],
    }


def _box_count(value: Any) -> int:
    if value is None:
        return 0
    if torch is not None and torch.is_tensor(value):
        if value.ndim != 2 or value.shape[-1] != 4:
            raise ValueError(f"boxes must have shape [N,4], got {tuple(value.shape)}")
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    raise TypeError(f"unsupported boxes value: {type(value)!r}")


_COLORMAP_STOPS = {
    "gray": (
        (0.0, (0, 0, 0)),
        (1.0, (255, 255, 255)),
    ),
    "inferno": (
        (0.0, (0, 0, 4)),
        (0.25, (87, 15, 109)),
        (0.5, (187, 55, 84)),
        (0.75, (249, 142, 8)),
        (1.0, (252, 255, 164)),
    ),
    "turbo": (
        (0.0, (48, 18, 59)),
        (0.2, (42, 120, 190)),
        (0.4, (20, 200, 160)),
        (0.6, (175, 230, 55)),
        (0.8, (250, 150, 30)),
        (1.0, (122, 4, 3)),
    ),
}


def _apply_colormap(values: np.ndarray, name: str) -> np.ndarray:
    try:
        stops = _COLORMAP_STOPS[name]
    except KeyError as error:
        raise ValueError(f"unsupported color map: {name!r}") from error
    positions = np.asarray([position for position, _ in stops], dtype=np.float32)
    colors = np.asarray([color for _, color in stops], dtype=np.float32)
    return np.stack(
        [np.interp(values, positions, colors[:, channel]) for channel in range(3)],
        axis=-1,
    ) / 255.0


def _scale_for_display(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0.0 or maximum > 1.0:
        if minimum >= 0.0 and maximum <= 255.0:
            array /= 255.0
        elif maximum > minimum:
            array = (array - minimum) / (maximum - minimum)
    return np.clip(array, 0.0, 1.0)


def _contrast_stretch(array: np.ndarray) -> np.ndarray:
    if array.shape[-1] == 1:
        low, high = np.percentile(array[..., 0], (1.0, 99.0))
        if high > low:
            array = (array - low) / (high - low)
    else:
        low = np.percentile(array, 1.0, axis=(0, 1), keepdims=True)
        high = np.percentile(array, 99.0, axis=(0, 1), keepdims=True)
        span = np.where(high > low, high - low, 1.0)
        array = (array - low) / span
    return np.clip(array, 0.0, 1.0)


def _image_from_value(
    value: Any,
    auto_contrast: bool = False,
    color_map: str = "inferno",
    display_scale: int = 1,
) -> Image.Image:
    if display_scale < 1:
        raise ValueError("display_scale must be positive")
    if isinstance(value, Image.Image):
        array = np.asarray(value)
    elif torch is not None and torch.is_tensor(value):
        array = value.detach().cpu().float().numpy()
    else:
        array = np.asarray(value)

    if array.ndim == 2:
        array = array[..., None]
    elif array.ndim == 3 and array.shape[0] <= 4 and array.shape[-1] > 4:
        array = np.moveaxis(array, 0, -1)
    if array.ndim != 3 or array.shape[-1] < 1:
        raise ValueError(f"image must be [H,W] or [C,H,W]/[H,W,C], got {array.shape}")

    array = _scale_for_display(array)
    if auto_contrast:
        array = _contrast_stretch(array)
    channels = array.shape[-1]
    if channels == 1:
        array = _apply_colormap(array[..., 0], color_map)
    elif channels == 2:
        array = np.concatenate([array, array[..., :1]], axis=-1)
    elif channels > 3:
        array = array[..., :3]
    image = Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")
    if display_scale > 1:
        image = image.resize(
            (image.width * display_scale, image.height * display_scale),
            Image.Resampling.LANCZOS,
        )
    return image



def _has_signal(image: Image.Image) -> bool:
    mask = ImageOps.grayscale(image).point(lambda value: 255 if value > 8 else 0)
    return mask.getbbox() is not None


def _overlay(source: Image.Image, condition: Image.Image) -> Image.Image:
    mask = ImageOps.grayscale(condition).point(lambda value: 255 if value > 8 else 0)
    return Image.composite(condition, source, mask)


def _label_image(image: Image.Image, label: str) -> Image.Image:
    image = image.convert("RGB")
    canvas = Image.new("RGB", (image.width, image.height + 22), "white")
    canvas.paste(image, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), label[:96], fill="black")
    return canvas


def _horizontal(images: Sequence[Image.Image], gap: int = 4) -> Image.Image:
    width = sum(image.width for image in images) + gap * max(0, len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width + gap
    return canvas


def _vertical(images: Sequence[Image.Image], gap: int = 4) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * max(0, len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    offset = 0
    for image in images:
        canvas.paste(image, (0, offset))
        offset += image.height + gap
    return canvas


def _tile(images: Sequence[Image.Image], columns: int, gap: int = 6) -> Image.Image:
    if not images:
        raise ValueError("cannot tile an empty image list")
    columns = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / columns)
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    canvas = Image.new(
        "RGB",
        (columns * cell_width + (columns - 1) * gap, rows * cell_height + (rows - 1) * gap),
        "white",
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas.paste(image, (column * (cell_width + gap), row * (cell_height + gap)))
    return canvas


def _finite_tensor(value: Any, name: str) -> None:
    if torch is not None and torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
        raise ValueError(f"batch field {name!r} contains non-finite values")


def validate_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Validate shapes, masks, IDs, timestamps, and frame ordering."""

    if not isinstance(batch, dict):
        raise TypeError(f"DataLoader batch must be a dict, got {type(batch)!r}")
    image_key = "vae_images" if "vae_images" in batch else "images"
    if image_key not in batch:
        raise KeyError("batch must contain vae_images or images")
    image_shape = _shape(batch[image_key])
    if len(image_shape) != 6:
        raise ValueError(f"{image_key} must have shape [B,T,V,C,H,W], got {image_shape}")
    batch_size, time_steps, view_count = image_shape[:3]
    if min(batch_size, time_steps, view_count) < 1:
        raise ValueError(f"empty batch dimensions: {image_shape}")
    _finite_tensor(batch[image_key], image_key)

    condition = batch.get("box_condition_images")
    if condition is not None:
        condition_shape = _shape(condition)
        if len(condition_shape) != 6 or condition_shape[:3] != (batch_size, time_steps, view_count):
            raise ValueError(
                "box_condition_images must share [B,T,V] with images, "
                f"got {condition_shape} versus {image_shape}"
            )
        _finite_tensor(condition, "box_condition_images")

    for name in ("bbox_available", "condition_valid", "pts"):
        value = batch.get(name)
        if value is not None:
            if _shape(value) != (batch_size, time_steps, view_count):
                raise ValueError(
                    f"{name} must have shape {(batch_size, time_steps, view_count)}, "
                    f"got {_shape(value)}"
                )
            _finite_tensor(value, name)

    sample_ids = batch.get("sample_ids")
    if sample_ids is None or _shape(sample_ids) != (batch_size, time_steps, view_count):
        raise ValueError(
            "sample_ids must have shape "
            f"{(batch_size, time_steps, view_count)}, got {_shape(sample_ids)}"
        )
    boxes = batch.get("boxes")
    sequence_values = batch.get("sequence")
    report_samples: list[dict[str, Any]] = []

    for batch_index in range(batch_size):
        first_id = parse_sample_id(str(_at(sample_ids, batch_index, 0, 0)))
        expected_sequence = (
            str(sequence_values[batch_index]) if sequence_values is not None else first_id["sequence"]
        )
        bbox_frames = 0
        total_boxes = 0
        for view_index in range(view_count):
            previous_frame: tuple[int, int | str] | None = None
            previous_pts: float | None = None
            expected_view: str | None = None
            for time_index in range(time_steps):
                metadata = parse_sample_id(str(_at(sample_ids, batch_index, time_index, view_index)))
                if metadata["dataset"] != first_id["dataset"]:
                    raise ValueError(f"sample {batch_index} changes dataset within a clip")
                if metadata["sequence"] != expected_sequence:
                    raise ValueError(f"sample {batch_index} changes sequence within a clip")
                if expected_view is None:
                    expected_view = metadata["view"]
                elif metadata["view"] != expected_view:
                    raise ValueError(f"sample {batch_index} changes view within a clip")
                current_frame = _frame_key(metadata["frame_id"])
                if previous_frame is not None and current_frame <= previous_frame:
                    raise ValueError(
                        f"sample {batch_index}, view {view_index} is not frame-ordered: "
                        f"{metadata['frame_id']!r}"
                    )
                previous_frame = current_frame

                if "pts" in batch:
                    current_pts = _scalar(_at(batch["pts"], batch_index, time_index, view_index))
                    if not math.isfinite(current_pts):
                        raise ValueError(f"sample {batch_index} contains a non-finite pts value")
                    if previous_pts is not None and current_pts < previous_pts - 1e-5:
                        raise ValueError(f"sample {batch_index}, view {view_index} has decreasing pts")
                    previous_pts = current_pts

                count = _box_count(_at(boxes, batch_index, time_index, view_index)) if boxes is not None else 0
                total_boxes += count
                available = (
                    _bool_at(batch["bbox_available"], batch_index, time_index, view_index)
                    if "bbox_available" in batch
                    else count > 0
                )
                valid = (
                    _bool_at(batch["condition_valid"], batch_index, time_index, view_index)
                    if "condition_valid" in batch
                    else count > 0
                )
                if boxes is not None and valid != bool(available and count > 0):
                    raise ValueError(
                        f"sample {batch_index}, frame {time_index}, view {view_index} "
                        f"has inconsistent bbox mask and count ({available=}, {valid=}, {count=})"
                    )
                if valid:
                    bbox_frames += 1
                    if condition is not None and not _has_signal(
                        _image_from_value(_at(condition, batch_index, time_index, view_index))
                    ):
                        raise ValueError(
                            f"sample {batch_index}, frame {time_index}, view {view_index} "
                            "is marked valid but its bbox condition is empty"
                        )

        report_samples.append(
            {
                "batch_index": batch_index,
                "dataset": first_id["dataset"],
                "sequence": expected_sequence,
                "frames": time_steps,
                "views": view_count,
                "bbox_frames": bbox_frames,
                "bbox_count": total_boxes,
            }
        )

    return {
        "image_key": image_key,
        "shape": image_shape,
        "condition_present": condition is not None,
        "boxes_present": boxes is not None,
        "samples": report_samples,
    }


def _frame_canvas(
    batch: dict[str, Any],
    batch_index: int,
    time_index: int,
    view_count: int,
    show_bbox: bool,
    image_key: str,
    color_map: str,
    display_scale: int,
) -> Image.Image:
    view_canvases: list[Image.Image] = []
    pts = (
        _scalar(_at(batch["pts"], batch_index, time_index, 0))
        if "pts" in batch
        else float(time_index)
    )
    for view_index in range(view_count):
        source = _image_from_value(
            _at(batch[image_key], batch_index, time_index, view_index),
            auto_contrast=True,
            color_map=color_map,
            display_scale=display_scale,
        )
        sample_id = str(_at(batch["sample_ids"], batch_index, time_index, view_index))
        panels = [_label_image(source, "image (contrast RGB)")]
        count = (
            _box_count(_at(batch["boxes"], batch_index, time_index, view_index))
            if "boxes" in batch
            else 0
        )
        valid = (
            _bool_at(batch["condition_valid"], batch_index, time_index, view_index)
            if "condition_valid" in batch
            else count > 0
        )
        if show_bbox:
            condition = _image_from_value(
                _at(batch["box_condition_images"], batch_index, time_index, view_index),
                display_scale=display_scale,
            )
            panels.extend(
                [
                    _label_image(condition, "bbox condition"),
                    _label_image(_overlay(source, condition), "image + bbox"),
                ]
            )
        view_body = _horizontal(panels)
        view_header = Image.new("RGB", (view_body.width, 22), "white")
        ImageDraw.Draw(view_header).text(
            (4, 4),
            f"view={view_index} bbox={count} valid={valid} {sample_id}"[:120],
            fill="black",
        )
        view_canvases.append(_vertical([view_header, view_body], gap=0))

    body = _vertical(view_canvases)
    dataset = str(batch.get("dataset", ["dataset"])[batch_index])
    sequence = str(batch.get("sequence", ["sequence"])[batch_index])
    header = Image.new("RGB", (body.width, 28), "white")
    ImageDraw.Draw(header).text(
        (4, 6),
        f"{dataset} | {sequence} | t={time_index} | pts={pts:.4f}",
        fill="black",
    )
    canvas = Image.new("RGB", (body.width, header.height + body.height), "white")
    canvas.paste(header, (0, 0))
    canvas.paste(body, (0, header.height))
    return canvas


def render_sample(
    batch: dict[str, Any],
    validation: dict[str, Any],
    batch_index: int,
    sample_index: int,
    output_dir: Path,
    gif_duration_ms: int,
    save_gif: bool,
    color_map: str = "inferno",
    display_scale: int = 2,
) -> dict[str, str]:
    image_key = str(validation["image_key"])
    _, time_steps, view_count = validation["shape"][:3]
    sample_report = validation["samples"][sample_index]
    show_bbox = bool(validation["condition_present"] and sample_report["bbox_frames"] > 0)
    frames = [
        _frame_canvas(
            batch,
            sample_index,
            time_index,
            view_count,
            show_bbox,
            image_key,
            color_map,
            display_scale,
        )
        for time_index in range(time_steps)
    ]
    stem = f"batch_{batch_index:04d}_sample_{sample_index:02d}"
    contact_path = output_dir / f"{stem}_contact.png"
    _tile(frames, columns=time_steps).save(contact_path)
    result = {"contact_sheet": str(contact_path)}
    if save_gif:
        gif_path = output_dir / f"{stem}.gif"
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=gif_duration_ms,
            loop=0,
            optimize=False,
        )
        result["gif"] = str(gif_path)
    return result

def render_batch_contact(
    rendered: Sequence[dict[str, str]],
    batch_index: int,
    output_dir: Path,
) -> str:
    """Stack rendered samples into one static view of a complete batch."""

    if not rendered:
        raise ValueError("cannot render an empty batch")
    contacts: list[Image.Image] = []
    for sample in rendered:
        with Image.open(sample["contact_sheet"]) as contact:
            contacts.append(contact.convert("RGB").copy())
    path = output_dir / f"batch_{batch_index:04d}_contact.png"
    _tile(contacts, columns=1).save(path)
    return str(path)

def inspect_batch(
    batch: dict[str, Any],
    batch_index: int,
    output_dir: Path,
    max_samples_per_batch: int = 0,
    gif_duration_ms: int = 180,
    save_gif: bool = True,
    color_map: str = "inferno",
    display_scale: int = 2,
) -> dict[str, Any]:
    """Validate and render one batch exactly as a training loop receives it."""

    if max_samples_per_batch < 0:
        raise ValueError("max_samples_per_batch must be non-negative")
    if gif_duration_ms < 1:
        raise ValueError("gif_duration_ms must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_batch(batch)
    sample_limit = min(
        len(validation["samples"]),
        max_samples_per_batch or len(validation["samples"]),
    )
    rendered = [
        render_sample(
            batch,
            validation,
            batch_index,
            sample_index,
            output_dir,
            gif_duration_ms,
            save_gif,
            color_map,
            display_scale,
        )
        for sample_index in range(sample_limit)
    ]
    batch_contact = render_batch_contact(rendered, batch_index, output_dir)
    return {
        "batch_index": batch_index,
        "validation": validation,
        "batch_contact": batch_contact,
        "rendered": rendered,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cfg", type=Path, help="dataset YAML/JSON config")
    source.add_argument("--task", choices=("bbox", "style", "joint"), help="multi task config")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/datasets"),
        help="directory containing multi task configs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataloader_visualization"))
    parser.add_argument("--batches", type=int, default=1, help="number of batches to inspect")
    parser.add_argument(
        "--max-samples-per-batch",
        type=int,
        default=0,
        help="maximum samples rendered per batch; 0 renders the complete batch",
    )
    parser.add_argument(
        "--color-map",
        choices=tuple(_COLORMAP_STOPS),
        default="inferno",
        help="single-channel display map: gray, inferno, or turbo",
    )
    parser.add_argument(
        "--display-scale",
        type=int,
        default=2,
        help="integer display upscaling factor applied after RGB conversion",
    )
    parser.add_argument("--gif-duration-ms", type=int, default=180)
    parser.add_argument("--no-gif", action="store_true", help="only save contact sheets")
    parser.add_argument("--set", nargs="*", default=[], help="dotted config overrides")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.batches < 1:
        raise SystemExit("--batches must be positive")
    if args.max_samples_per_batch < 0:
        raise SystemExit("--max-samples-per-batch must be non-negative")
    if args.display_scale < 1:
        raise SystemExit("--display-scale must be positive")
    if torch is None:
        raise SystemExit("visualizing DataLoader batches requires PyTorch")

    if args.task is not None:
        loader, _ = load_task_dataloader(args.task, config_dir=args.config_dir, overrides=args.set)
    else:
        loader, _ = load_dataloader_from_config(args.cfg, overrides=args.set)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    iterator = iter(loader)
    batch_reports: list[dict[str, Any]] = []
    for batch_index in range(args.batches):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_report = inspect_batch(
            batch,
            batch_index,
            args.output_dir,
            args.max_samples_per_batch,
            args.gif_duration_ms,
            not args.no_gif,
            args.color_map,
            args.display_scale,
        )
        batch_reports.append(batch_report)
        print(json.dumps(batch_report, ensure_ascii=False), flush=True)

    if not batch_reports:
        raise SystemExit("DataLoader produced no batches")
    report = {
        "batches_requested": args.batches,
        "batches_rendered": len(batch_reports),
        "output_dir": str(args.output_dir.resolve()),
        "batches": batch_reports,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "batches": len(batch_reports)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# 文件讲解：
# 1. 该脚本直接消费配置工厂返回的 PyTorch DataLoader；每次 next(iterator)
#    得到一个 batch dict，而不是重新访问底层 dataset。
# 2. validate_batch 检查 [B,T,V] 对齐、sample_id 的 dataset/sequence/view 稳定性、
#    frame id 顺序、pts 单调性，以及 bbox mask、boxes 和 condition 的一致性。
# 3. 每个 batch 保存一个 batch contact sheet；同时按 sample 保存时间 contact
#    sheet 和 GIF。单通道 source 默认做 1%/99% 对比度拉伸和 inferno RGB 映射，
#    display_scale 默认 2；这些只影响显示，不改变 batch tensor。
# 4. 仅在当前 clip 有有效框时额外显示 condition 与 image+bbox 叠加结果；
#    style 模式没有 box_condition_images，因此只显示连续原图。
# 5. inspect_batch 可直接放进训练循环，在 training_step 前检查有限数量的 batch；
#    校验失败立即抛出异常，避免错误数据静默进入训练。
# 6. 运行示例：
#    PYTHONPATH=src python scripts/prepare/visualize_dataset_batch.py \
#      --task joint --batches 2 --output-dir outputs/joint_visual \
#      --max-samples-per-batch 0
