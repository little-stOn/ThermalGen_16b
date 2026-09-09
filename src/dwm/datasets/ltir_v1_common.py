"""LTIR v1 sequence and ground-truth parsing."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Iterable

from PIL import Image

from dwm.datasets.common import (
    BBoxFrameRecord,
    BBoxViewRecord,
    limit_sequence,
    make_bbox_annotation,
)


_LABELS = {
    "birds": "animal.bird",
    "car": "vehicle.car",
    "crossing": "human.pedestrian",
    "crouching": "human.pedestrian",
    "crowd": "human.pedestrian",
    "depthwise_crossing": "human.pedestrian",
    "garden": "human.pedestrian",
    "hiding": "human.pedestrian",
    "horse": "animal.horse",
    "jacket": "human.pedestrian",
    "mixed_distractors": "human.pedestrian",
    "quadrocopter": "vehicle.quadrocopter",
    "quadrocopter2": "vehicle.quadrocopter",
    "rhino_behind_tree": "animal.rhino",
    "running_rhino": "animal.rhino",
    "saturated": "human.pedestrian",
    "selma": "human.pedestrian",
    "soccer": "human.pedestrian",
    "street": "human.pedestrian",
    "trees": "human.pedestrian",
}


def resolve_ltir_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if (root / "ltir_v1_0_8bit_16bit").is_dir():
        root = root / "ltir_v1_0_8bit_16bit"
    if not root.is_dir():
        raise FileNotFoundError(f"LTIR root does not exist: {root}")
    return root


def _sequence_label(name: str) -> str:
    key = re.sub(r"^\d+_", "", name.lower())
    return _LABELS.get(key, f"ltir.{key or 'unknown'}")


def parse_groundtruth(path: str | Path) -> tuple[tuple[float, float, float, float], ...]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"LTIR groundtruth file does not exist: {path}")
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            raise ValueError(f"LTIR groundtruth contains an empty row: {path}:{line_number}")
        tokens = [token for token in re.split(r"[;,\s]+", line) if token]
        values = tuple(float(token) for token in tokens)
        if len(values) == 4:
            x1, y1, x2, y2 = values
        elif len(values) == 8:
            xs = values[0::2]
            ys = values[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            raise ValueError(f"LTIR groundtruth row must contain 4 or 8 values: {path}:{line_number}")
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        raise ValueError(f"LTIR groundtruth file is empty: {path}")
    return tuple(boxes)


def _image_files(sequence_root: Path) -> list[Path]:
    images = sorted(
        [path for path in sequence_root.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda path: (int(path.stem) if path.stem.isdigit() else path.stem, path.name),
    )
    if not images:
        raise ValueError(f"LTIR sequence has no image files: {sequence_root}")
    return images


def load_ltir_records(
    dataset_root: str | Path,
    mode: str = "16bit",
    sequence_names: Iterable[str] | None = None,
    load_annotations: bool = True,
    max_frames_per_sequence: int | None = None,
    source_fps: float = 25.0,
) -> tuple[BBoxFrameRecord, ...]:
    root = resolve_ltir_root(dataset_root)
    mode = str(mode).lower()
    if mode not in {"8bit", "16bit", "all"}:
        raise ValueError("LTIR mode must be 8bit, 16bit, or all")
    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("source_fps must be finite and positive")
    allowed = None if sequence_names is None else {str(value) for value in sequence_names}
    records: list[BBoxFrameRecord] = []
    for sequence_root in sorted(path for path in root.iterdir() if path.is_dir()):
        name = sequence_root.name
        if allowed is not None and name not in allowed:
            continue
        if mode != "all" and not name.startswith(f"{mode.split('bit')[0]}_"):
            continue
        images = _image_files(sequence_root)
        if load_annotations:
            groundtruth = parse_groundtruth(sequence_root / "groundtruth.txt")
            if len(images) != len(groundtruth):
                raise ValueError(
                    f"LTIR image/groundtruth count mismatch for {name}: {len(images)} != {len(groundtruth)}"
                )
        else:
            groundtruth = (None,) * len(images)
        selected = limit_sequence(
            tuple(enumerate(zip(images, groundtruth, strict=True), 1)),
            max_frames_per_sequence,
        )
        label = _sequence_label(name)
        with Image.open(selected[0][1][0]) as image:
            width, height = image.size
        for index, (image_path, box) in selected:
            boxes = ()
            if box is not None:
                x1, y1, x2, y2 = box
                x1, x2 = max(0.0, min(x1, x2)), min(float(width), max(x1, x2))
                y1, y2 = max(0.0, min(y1, y2)), min(float(height), max(y1, y2))
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"LTIR invalid clipped bbox in {image_path}: {box!r}")
                boxes = (
                    make_bbox_annotation((x1, y1, x2, y2), label, track_id=f"{name}:0"),
                )
            records.append(
                BBoxFrameRecord(
                    sequence=name,
                    frame_id=f"{index:08d}",
                    timestamp=(index - 1) / source_fps,
                    views=(
                        BBoxViewRecord(
                            "thermal",
                            image_path,
                            boxes,
                            annotations_available=load_annotations,
                        ),
                    ),
                    metadata={"sequence_name": name, "mode": name.split("_", 1)[0]},
                )
            )
    if not records:
        raise ValueError(f"LTIR selection produced no sequences: mode={mode!r}")
    return tuple(records)

# 文件讲解：
# 1. resolve_ltir_root 处理外层 ltir_v1 和内部
#    ltir_v1_0_8bit_16bit 两种传入方式；load_ltir_records 再按 mode
#    选择 8bit、16bit 或全部序列。
# 2. parse_groundtruth 支持每行四个 xyxy 值或八个角点值，并统一成正面积
#    bbox。每个序列只打开第一帧获取宽高，后续帧只建立路径和标注对象。
# 3. 帧名按数字排序，groundtruth 行数必须和图像数完全相等；任何错位都会
#    立即报错，避免训练时图像和框错配。
# 4. 该文件不创建 DataLoader、不做 tensor transform；它的唯一职责是把
#    LTIR 文件格式转换成 shared BBoxFrameRecord。
