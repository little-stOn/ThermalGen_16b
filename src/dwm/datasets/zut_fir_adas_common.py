"""ZUT-FIR-ADAS recording and YOLO annotation parsing."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import pickle
import re
from typing import Iterable, Sequence

from PIL import Image
import numpy as np

from dwm.datasets.common import (
    BBoxFrameRecord,
    BBoxViewRecord,
    limit_sequence,
    make_bbox_annotation,
)


DEFAULT_CLASS_NAMES = (
    "human.pedestrian",
    "human.pedestrian.occluded",
    "human.body_part",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.scooter",
    "unknown",
    "human.baby_carriage",
    "animal",
)


def resolve_zut_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "zut_fir_adas" and root.is_dir():
        return root
    nested = root / "zut_fir_adas"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"ZUT-FIR-ADAS root does not exist: {root}")


def _split_bucket(sequence: str) -> int:
    return int(hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:8], 16) % 10

def _in_split(sequence: str, split: str | None) -> bool:
    if split is None or str(split).lower() in {"all", "none"}:
        return True
    recording = sequence.rsplit("/", 1)[-1]
    is_benchmark = recording.endswith("_b")
    split = str(split).lower()
    if split in {"test", "benchmark"}:
        return is_benchmark
    if split == "train":
        return not is_benchmark and _split_bucket(sequence) != 0
    if split in {"val", "validation"}:
        return not is_benchmark and _split_bucket(sequence) == 0
    raise ValueError("ZUT split must be train, val, test, benchmark, or all")


def parse_yolo_annotations(
    path: str | Path,
    image_size: tuple[int, int],
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    class_whitelist: Iterable[str] | None = None,
) -> tuple:
    path = Path(path)
    if not path.is_file():
        return ()
    width, height = (float(value) for value in image_size)
    whitelist = None if class_whitelist is None else {str(value) for value in class_whitelist}
    annotations = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        tokens = raw_line.split()
        if not tokens:
            continue
        if len(tokens) not in {5, 6}:
            raise ValueError(f"ZUT YOLO row must contain 5 or 6 values: {path}:{line_number}")
        class_id = int(tokens[0])
        cx, cy, box_width, box_height = (float(value) for value in tokens[1:5])
        if not np.all(np.isfinite([cx, cy, box_width, box_height])):
            raise ValueError(f"ZUT YOLO row contains non-finite values: {path}:{line_number}")
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        label = class_names[class_id] if 0 <= class_id < len(class_names) else f"zut.class_{class_id}"
        if whitelist is not None and label not in whitelist:
            continue
        x1 = max(0.0, (cx - box_width / 2.0) * width)
        y1 = max(0.0, (cy - box_height / 2.0) * height)
        x2 = min(width, (cx + box_width / 2.0) * width)
        y2 = min(height, (cy + box_height / 2.0) * height)
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(tokens[5]) if len(tokens) == 6 else None
        annotations.append(
            make_bbox_annotation(
                (x1, y1, x2, y2), label,
                track_id=f"{path.stem}:{len(annotations)}", score=score
            )
        )
    return tuple(annotations)


def _frame_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"frameIndex_(\d+)", path.stem)
    return (int(match.group(1)) if match else 0, path.name)

def _default_index_cache(root: Path, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return root / ".dwm_cache" / f"zut_{digest}.pkl"


def load_zut_records(
    dataset_root: str | Path,
    frame_directory: str = "16BitFrames",
    split: str | None = "all",
    countries: Iterable[str] | None = None,
    routes: Iterable[str] | None = None,
    recordings: Iterable[str] | None = None,
    include_empty_frames: bool = True,
    class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
    class_whitelist: Iterable[str] | None = None,
    index_cache: str | Path | None = None,
    use_index_cache: bool = True,
    load_annotations: bool = True,
    max_frames_per_sequence: int | None = None,
    source_fps: float = 25.0,
) -> tuple[BBoxFrameRecord, ...]:
    root = resolve_zut_root(dataset_root)
    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("source_fps must be finite and positive")
    selected_countries = None if countries is None else tuple(sorted(str(value) for value in countries))
    selected_routes = None if routes is None else tuple(sorted(str(value) for value in routes))
    selected_recordings = None if recordings is None else tuple(sorted(str(value) for value in recordings))
    normalized_class_names = tuple(str(value) for value in class_names)
    normalized_whitelist = (
        None
        if class_whitelist is None
        else tuple(sorted(str(value) for value in class_whitelist))
    )
    cache_key = repr(
        (
            "zut-index-v6",
            str(root),
            frame_directory,
            split,
            selected_countries,
            selected_routes,
            selected_recordings,
            include_empty_frames,
            normalized_class_names,
            normalized_whitelist,
            load_annotations,
            max_frames_per_sequence,
            source_fps,
        )
    )
    cache_path = (
        _default_index_cache(root, cache_key)
        if index_cache is None
        else Path(index_cache).expanduser().resolve()
    )
    if use_index_cache:
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("key") == cache_key:
                return tuple(payload["records"])
        except (OSError, EOFError, KeyError, AttributeError, pickle.UnpicklingError):
            pass

    countries_set = None if selected_countries is None else set(selected_countries)
    routes_set = None if selected_routes is None else set(selected_routes)
    recordings_set = None if selected_recordings is None else set(selected_recordings)
    whitelist_set = None if normalized_whitelist is None else set(normalized_whitelist)
    records: list[BBoxFrameRecord] = []
    for country_root in sorted(path for path in root.iterdir() if path.is_dir()):
        if countries_set is not None and country_root.name not in countries_set:
            continue
        for route_root in sorted(path for path in country_root.iterdir() if path.is_dir()):
            if routes_set is not None and route_root.name not in routes_set:
                continue
            for recording_root in sorted(path for path in route_root.iterdir() if path.is_dir()):
                if recordings_set is not None and recording_root.name not in recordings_set:
                    continue
                sequence = f"{country_root.name}/{route_root.name}/{recording_root.name}"
                if not _in_split(sequence, split):
                    continue
                frame_root = recording_root / frame_directory
                annotation_root = recording_root / "annotations"
                if not frame_root.is_dir():
                    continue
                frames = sorted(
                    [
                        path
                        for path in frame_root.iterdir()
                        if path.is_file() and path.suffix.lower() == ".png"
                    ],
                    key=_frame_sort_key,
                )
                if not frames:
                    continue
                selected_frames = limit_sequence(
                    tuple(enumerate(frames)),
                    max_frames_per_sequence,
                )
                with Image.open(selected_frames[0][1]) as image:
                    image_size = image.size
                for original_index, frame_path in selected_frames:
                    annotations_available = load_annotations and annotation_root.is_dir()
                    annotations = (
                        parse_yolo_annotations(
                            annotation_root / f"{frame_path.stem}.txt",
                            image_size,
                            normalized_class_names,
                            whitelist_set,
                        )
                        if annotations_available
                        else ()
                    )
                    if annotations_available and not include_empty_frames and not annotations:
                        continue
                    records.append(
                        BBoxFrameRecord(
                            sequence=sequence,
                            frame_id=frame_path.stem,
                            timestamp=original_index / source_fps,
                            views=(
                                BBoxViewRecord(
                                    "thermal",
                                    frame_path,
                                    annotations,
                                    annotations_available=annotations_available,
                                ),
                            ),
                            metadata={
                                "country": country_root.name,
                                "route": route_root.name,
                                "recording": recording_root.name,
                                "frame_directory": frame_directory,
                            },
                        )
                    )
    if not records:
        raise ValueError("ZUT selection produced no frame records")
    result = tuple(records)
    if use_index_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary_path.open("wb") as handle:
                pickle.dump({"key": cache_key, "records": result}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_path.replace(cache_path)
        except OSError:
            pass
    return result

# 文件讲解：
# 1. 该解析器把 ZUT 的 recording 目录和 annotations/*.txt 对齐为
#    BBoxFrameRecord；每条 YOLO 的 cx/cy/w/h 会结合第一帧尺寸转换为绝对 xyxy。
# 2. class_names 与 class_whitelist 分别控制类别映射和保留类别；非法行、
#    非有限数值、负面积框都会被拒绝或跳过，不能静默制造伪框。
# 3. _in_split 使用 recording 的 _b 后缀区分 benchmark；因此 split 选择发生在
#    clip 构造之前，不会把 benchmark 与 train 混在一起。
# 4. index cache 的 key 包含数据根、帧目录、split、子集和类别选择；缓存内容
#    是 pickle 化的路径/标注索引，不包含图像，适合大规模 TXT 目录的重复调试。
