"""FLIR ADAS v2 COCO indexing for bbox-aware temporal loading."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from dwm.datasets.common import BBoxAnnotation, BBoxFrameRecord, BBoxViewRecord, make_bbox_annotation


_FRAME_RE = re.compile(r"^video-(?P<video>.+)-frame-(?P<frame>\d+)-")
_SPLIT_DIRS = {
    ("train", "rgb"): "images_rgb_train",
    ("val", "rgb"): "images_rgb_val",
    ("test", "rgb"): "video_rgb_test",
    ("train", "thermal"): "images_thermal_train",
    ("val", "thermal"): "images_thermal_val",
    ("test", "thermal"): "video_thermal_test",
}
_CATEGORY_ALIASES = {
    "person": "human.pedestrian",
    "rider": "human.rider",
    "bike": "vehicle.bicycle",
    "car": "vehicle.car",
    "motor": "vehicle.motorcycle",
    "bus": "vehicle.bus",
    "truck": "vehicle.truck",
}


def resolve_flir_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "FLIR_ADAS_v2" and root.is_dir():
        return root
    nested = root / "FLIR_ADAS_v2"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"FLIR ADAS v2 root does not exist: {root}")


def _split_directory(root: Path, split: str, modality: str) -> Path:
    split_name = str(split).lower()
    modality_name = str(modality).lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError("FLIR split must be train, val, or test")
    if modality_name not in {"rgb", "thermal"}:
        raise ValueError("FLIR modality must be rgb or thermal")
    return root / _SPLIT_DIRS[(split_name, modality_name)]


def _canonical_label(name: Any) -> str:
    raw = str(name).strip().lower()
    if raw in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[raw]
    normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return f"flir.{normalized or 'unknown'}"


def _frame_identity(image: dict[str, Any]) -> tuple[str, int]:
    file_name = Path(str(image.get("file_name", ""))).name
    match = _FRAME_RE.match(file_name)
    extra = image.get("extra_info") or {}
    if match is not None:
        return str(extra.get("video_id") or match.group("video")), int(match.group("frame"))
    return str(extra.get("video_id") or "sequence"), int(image.get("id", 0))


def _image_path(split_root: Path, file_name: str) -> Path:
    relative = Path(file_name)
    path = relative if relative.is_absolute() else split_root / relative
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FLIR image referenced by COCO is missing: {path}")
    return path


def _load_coco_records(
    split_root: Path,
    category_whitelist: set[str] | None = None,
    load_annotations: bool = True,
) -> tuple[BBoxFrameRecord, ...]:
    annotation_path = split_root / "coco.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"FLIR COCO annotation file does not exist: {annotation_path}")
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = (
        {
            int(item["id"]): _canonical_label(item.get("name", "unknown"))
            for item in payload.get("categories", [])
        }
        if load_annotations
        else {}
    )
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if load_annotations:
        for annotation in payload.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    records: list[BBoxFrameRecord] = []
    for image in sorted(
        payload.get("images", []),
        key=lambda value: (*_frame_identity(value), int(value.get("id", 0))),
    ):
        video_id, frame_number = _frame_identity(image)
        boxes: list[BBoxAnnotation] = []
        for annotation in annotations_by_image.get(int(image["id"]), []):
            label = categories.get(int(annotation.get("category_id", -1)), "flir.unknown")
            if category_whitelist is not None and label not in category_whitelist:
                continue
            raw_box = annotation.get("bbox")
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                raise ValueError(f"FLIR COCO bbox must be [x,y,width,height]: {raw_box!r}")
            x, y, width, height = (float(value) for value in raw_box)
            if width <= 0.0 or height <= 0.0:
                continue
            boxes.append(
                make_bbox_annotation(
                    (x, y, x + width, y + height),
                    label,
                    track_id=f"ann-{annotation.get('id', len(boxes))}",
                )
            )
        records.append(
            BBoxFrameRecord(
                sequence=f"{split_root.name}:{video_id}",
                frame_id=f"{frame_number:06d}",
                views=(
                    BBoxViewRecord(
                        name="thermal" if "thermal" in split_root.name else "rgb",
                        path=_image_path(split_root, str(image["file_name"])),
                        boxes=tuple(boxes),
                        annotations_available=load_annotations,
                    ),
                ),
                metadata={
                    "image_id": int(image["id"]),
                    "file_name": str(image["file_name"]),
                    "video_id": video_id,
                    "frame_number": frame_number,
                    "width": int(image.get("width", 0)),
                    "height": int(image.get("height", 0)),
                },
            )
        )
    if not records:
        raise ValueError(f"FLIR COCO split has no images: {split_root}")
    return tuple(records)


def load_flir_records(
    dataset_root: str | Path,
    split: str = "train",
    modality: str = "thermal",
    view_mode: str = "single",
    category_whitelist: Iterable[str] | None = None,
    load_annotations: bool = True,
) -> tuple[BBoxFrameRecord, ...]:
    """Load FLIR frames, optionally pairing the official video-test views."""

    root = resolve_flir_root(dataset_root)
    whitelist = None if category_whitelist is None else {str(value) for value in category_whitelist}
    if view_mode == "single":
        split_root = _split_directory(root, split, modality)
        return _load_coco_records(split_root, whitelist, load_annotations)
    if view_mode != "multiview":
        raise ValueError("FLIR view_mode must be single or multiview")
    if str(split).lower() != "test":
        raise ValueError("FLIR multiview pairing is only defined for video_test")
    rgb_root = _split_directory(root, "test", "rgb")
    thermal_root = _split_directory(root, "test", "thermal")
    rgb_records = _load_coco_records(rgb_root, whitelist, load_annotations)
    thermal_records = _load_coco_records(thermal_root, whitelist, load_annotations)
    mapping_path = root.parent / "rgb_to_thermal_vid_map.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"FLIR RGB-to-thermal map does not exist: {mapping_path}")
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    thermal_by_file = {
        str((record.metadata or {}).get("file_name")): record for record in thermal_records
    }
    thermal_by_file.update(
        {Path(key).name: value for key, value in list(thermal_by_file.items())}
    )
    paired: list[BBoxFrameRecord] = []
    for rgb_record in rgb_records:
        rgb_file = str((rgb_record.metadata or {}).get("file_name"))
        thermal_file = mapping.get(Path(rgb_file).name, mapping.get(rgb_file))
        thermal_record = thermal_by_file.get(str(thermal_file))
        if thermal_record is None:
            continue
        paired.append(
            BBoxFrameRecord(
                sequence=rgb_record.sequence,
                frame_id=rgb_record.frame_id,
                views=(rgb_record.views[0], thermal_record.views[0]),
                timestamp=rgb_record.timestamp,
                metadata={"rgb": rgb_record.metadata, "thermal": thermal_record.metadata},
            )
        )
    if not paired:
        raise ValueError("FLIR video-test RGB and thermal COCO sets have no paired frames")
    return tuple(paired)

# 文件讲解：
# 1. resolve_flir_root 允许传入 flir 根目录或 FLIR_ADAS_v2 子目录；
#    _split_directory 再把 train/val/test 与 rgb/thermal 映射到实际目录。
# 2. _load_coco_records 一次读取 coco.json，建立 image_id 到 annotation 的
#    索引，并按 video_id、frame number 排序。初始化阶段只保存路径和小型
#    BBoxAnnotation，不读取图像像素，因此随机取样不会重复扫描 COCO。
# 3. COCO bbox 被严格校验为正面积 xyxy；未知类别保留为 flir.<name>，
#    不会被静默改成错误类别。空帧由上层 bbox_policy 决定是否进入 clip。
# 4. multiview 只使用官方 test 映射文件逐帧配对 RGB/thermal，配对失败时
#    报错或明确跳过，禁止按排序位置硬拼两个不同视频。
# 5. 维护重点：如果新增 FLIR 子集，先补 _SPLIT_DIRS 和 COCO 路径规则，
#    再用 flir.yaml 验证输出的 [T][V] 和 condition 形状。
