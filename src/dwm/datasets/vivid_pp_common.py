"""VIVID++ generated annotation manifest parsing."""

from __future__ import annotations

from collections import defaultdict
import bisect
import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
import pickle
import sys
from typing import Any, Iterable, Sequence

from PIL import Image
from dwm.datasets.common import BBoxAnnotation, BBoxFrameRecord, BBoxViewRecord, make_bbox_annotation


_DEFAULT_BOX_KEYS = (
    "bbox_thr_lidar_xyxy",
    "bbox_thr_geometry_xyxy",
    "bbox_thr_unexpanded_xyxy",
    "bbox_thr_xyxy",
)


def resolve_vivid_annotation_root(
    annotation_root: str | Path,
    require_annotations: bool = True,
) -> Path:
    root = Path(annotation_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"VIVID++ annotation root does not exist: {root}")
    if not (root / "frames.jsonl").is_file():
        raise FileNotFoundError(f"VIVID++ annotation root must contain frames.jsonl: {root}")
    if require_annotations and not (root / "detections.jsonl").is_file():
        raise FileNotFoundError(
            f"VIVID++ annotation root must contain detections.jsonl: {root}"
        )
    return root

def _default_annotation_cache(root: Path, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return root / ".dwm_cache" / f"vivid_annotation_{digest}.pkl"


def _resolve_image(annotation_root: Path, value: str) -> Path:
    path = Path(value)
    candidates = [path if path.is_absolute() else annotation_root / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"VIVID++ generated image does not exist: {value}")


def _frame_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record.get("bag", "")),
        str(record.get("thermal_sequence", record.get("thermal_path", ""))),
        str(record.get("thermal_path", "")),
    )


def _timestamp(record: dict) -> float | None:
    value = record.get("thermal_stamp", record.get("rgb_stamp"))
    return None if value is None else float(value)

def resolve_vivid_raw_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "vivid_pp" and root.parent.name == "raw" and root.is_dir():
        return root
    nested = root / "vivid_pp"
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"VIVID++ raw root does not exist: {root}")


@lru_cache(maxsize=1)
def _load_bag_reader_module(project_root: str) -> Any:
    module_path = Path(project_root) / "scripts" / "prepare" / "vivid_pp_calibration_pilot.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"VIVID++ bag reader module does not exist: {module_path}")
    module_name = "_dwm_vivid_pp_bag_reader"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load VIVID++ bag reader module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _raw_cache_path(
    cache_dir: Path,
    bag_path: Path,
    mode: str,
    modality: str,
    view_mode: str,
    max_frames_per_bag: int | None,
    synchronization_tolerance_ms: float,
) -> tuple[Path, str]:
    key = repr(
        (
            "vivid-raw-v1",
            str(bag_path),
            bag_path.stat().st_size,
            bag_path.stat().st_mtime_ns,
            mode,
            modality,
            view_mode,
            max_frames_per_bag,
            float(synchronization_tolerance_ms),
        )
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{digest}.pkl", key


def _nearest_raw_sample(
    timestamp: float,
    candidates: Sequence[tuple[float, Path, int]],
) -> tuple[int, float] | None:
    if not candidates:
        return None
    timestamps = [item[0] for item in candidates]
    position = bisect.bisect_left(timestamps, timestamp)
    candidate_indices = range(max(0, position - 1), min(len(candidates), position + 1))
    index = min(candidate_indices, key=lambda value: abs(candidates[value][0] - timestamp))
    return index, abs(candidates[index][0] - timestamp)


def _extract_raw_bag_records(
    bag_path: Path,
    subset: str,
    mode: str,
    modality: str,
    view_mode: str,
    max_frames_per_bag: int | None,
    synchronization_tolerance_ms: float,
    cache_dir: Path,
    use_cache: bool,
) -> tuple[BBoxFrameRecord, ...]:
    project_root = bag_path.parents[4]
    cache_path, cache_key = _raw_cache_path(
        cache_dir,
        bag_path,
        mode,
        modality,
        view_mode,
        max_frames_per_bag,
        synchronization_tolerance_ms,
    )
    if use_cache:
        try:
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("key") == cache_key and all(
                view.path.is_file()
                for record in payload["records"]
                for view in record.views
            ):
                return tuple(payload["records"])
        except (OSError, EOFError, KeyError, AttributeError, pickle.UnpicklingError):
            pass

    module = _load_bag_reader_module(str(project_root))
    topics = module._topic_set(mode)
    selected_topics = {
        "rgb": topics["rgb"],
        "thermal": topics["thermal"],
    }
    if modality not in {"rgb", "thermal"}:
        raise ValueError("VIVID++ modality must be rgb or thermal")
    if view_mode not in {"single", "multiview"}:
        raise ValueError("VIVID++ view_mode must be single or multiview")
    if max_frames_per_bag is not None and max_frames_per_bag < 1:
        raise ValueError("max_frames_per_bag must be positive")

    raw_dir = cache_dir / f"{cache_path.stem}_{bag_path.stem}"
    rgb_samples: list[tuple[float, Path, int]] = []
    thermal_samples: list[tuple[float, Path, int]] = []
    reader = module.RosBagReader(bag_path)
    for message in reader.messages():
        if view_mode == "single" and message.topic != selected_topics[modality]:
            continue
        if message.topic == selected_topics["thermal"]:
            stream = thermal_samples
            kind = "thermal"
        elif message.topic == selected_topics["rgb"]:
            stream = rgb_samples
            kind = "rgb"
        else:
            continue
        if max_frames_per_bag is not None and len(stream) >= max_frames_per_bag:
            if view_mode == "single" or (
                len(rgb_samples) >= max_frames_per_bag
                and len(thermal_samples) >= max_frames_per_bag
            ):
                break
            continue
        sample = module._read_image_message(message.topic, message.payload)
        path = raw_dir / kind / f"{len(stream):08d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(sample.array).save(path)
        stream.append((float(sample.stamp), path, int(sample.sequence)))

    sequence = f"{subset}:{bag_path.stem}"
    records: list[BBoxFrameRecord] = []
    if view_mode == "single":
        selected = thermal_samples if modality == "thermal" else rgb_samples
        for index, (timestamp, path, source_sequence) in enumerate(selected):
            records.append(
                BBoxFrameRecord(
                    sequence=sequence,
                    frame_id=f"{source_sequence:08d}",
                    timestamp=timestamp,
                    views=(BBoxViewRecord(modality, path, (), annotations_available=False),),
                    metadata={"subset": subset, "bag": str(bag_path), "source_index": index},
                )
            )
    else:
        thermal_samples.sort(key=lambda item: item[0])
        rgb_samples.sort(key=lambda item: item[0])
        max_delta = float(synchronization_tolerance_ms) / 1000.0
        for index, (rgb_time, rgb_path, rgb_sequence) in enumerate(rgb_samples):
            match = _nearest_raw_sample(rgb_time, thermal_samples)
            if match is None or match[1] > max_delta:
                continue
            thermal_index, _delta = match
            thermal_time, thermal_path, thermal_sequence = thermal_samples[thermal_index]
            records.append(
                BBoxFrameRecord(
                    sequence=sequence,
                    frame_id=f"{thermal_sequence:08d}",
                    timestamp=thermal_time,
                    views=(
                        BBoxViewRecord("rgb", rgb_path, (), annotations_available=False),
                        BBoxViewRecord("thermal", thermal_path, (), annotations_available=False),
                    ),
                    metadata={
                        "subset": subset,
                        "bag": str(bag_path),
                        "rgb_sequence": rgb_sequence,
                        "thermal_sequence": thermal_sequence,
                        "sync_delta_ms": abs(rgb_time - thermal_time) * 1000.0,
                        "source_index": index,
                    },
                )
            )
    if not records:
        raise ValueError(f"VIVID++ raw bag produced no frames: {bag_path}")
    result = tuple(records)
    if use_cache:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
            with temporary_path.open("wb") as handle:
                pickle.dump({"key": cache_key, "records": result}, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_path.replace(cache_path)
        except OSError:
            pass
    return result


def load_vivid_raw_records(
    dataset_root: str | Path,
    subsets: Iterable[str] | None = None,
    modality: str = "thermal",
    view_mode: str = "single",
    max_frames_per_bag: int | None = None,
    synchronization_tolerance_ms: float = 50.0,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> tuple[BBoxFrameRecord, ...]:
    root = resolve_vivid_raw_root(dataset_root)
    selected = None if subsets is None else {str(value) for value in subsets}
    cache_root = (
        root / ".dwm_cache" / "vivid_raw"
        if cache_dir is None
        else Path(cache_dir).expanduser().resolve()
    )
    records: list[BBoxFrameRecord] = []
    for subset_root in sorted(path for path in root.iterdir() if path.is_dir()):
        if selected is not None and subset_root.name not in selected:
            continue
        if subset_root.name.startswith("driving"):
            mode = "driving"
        elif subset_root.name.startswith("handheld"):
            mode = "handheld"
        else:
            continue
        for bag_path in sorted(subset_root.glob("*.bag")):
            records.extend(
                _extract_raw_bag_records(
                    bag_path,
                    subset_root.name,
                    mode,
                    modality,
                    view_mode,
                    max_frames_per_bag,
                    synchronization_tolerance_ms,
                    cache_root,
                    use_cache,
                )
            )
    if not records:
        raise ValueError("VIVID++ raw selection produced no image records")
    return tuple(records)


def merge_vivid_annotations(
    raw_records: Sequence[BBoxFrameRecord],
    annotated_records: Sequence[BBoxFrameRecord],
) -> tuple[BBoxFrameRecord, ...]:
    annotations = {
        ((record.metadata or {}).get("bag"), record.frame_id): record
        for record in annotated_records
    }
    merged: list[BBoxFrameRecord] = []
    for record in raw_records:
        annotation = annotations.get(((record.metadata or {}).get("bag"), record.frame_id))
        if annotation is None:
            merged.append(record)
            continue
        annotation_by_view = {view.name: view for view in annotation.views}
        views = tuple(
            BBoxViewRecord(
                view.name,
                view.path,
                annotation_by_view.get(view.name, view).boxes,
                annotations_available=view.name in annotation_by_view,
            )
            for view in record.views
        )
        merged.append(
            BBoxFrameRecord(
                sequence=record.sequence,
                frame_id=record.frame_id,
                timestamp=record.timestamp,
                views=views,
                metadata=record.metadata,
            )
        )
    return tuple(merged)


def _box_from_record(
    record: dict,
    box_keys: Iterable[str],
    min_score: float,
    class_whitelist: set[str] | None,
) -> BBoxAnnotation | None:
    label = str(record.get("class_name", record.get("detector_label", "vehicle.unknown")))
    if class_whitelist is not None and label not in class_whitelist:
        return None
    score = record.get("score")
    score = None if score is None else float(score)
    if score is not None and score < min_score:
        return None
    raw_box = next((record.get(key) for key in box_keys if record.get(key) is not None), None)
    if raw_box is None:
        return None
    track_id = record.get("track_id", record.get("instance_id"))
    if track_id is None:
        track_id = f"det-{record.get('rgb_sequence', record.get('thermal_sequence', 'unknown'))}"
    return make_bbox_annotation(raw_box, label, str(track_id), score)


def load_vivid_records(
    annotation_root: str | Path,
    modality: str = "thermal",
    view_mode: str = "single",
    min_score: float = 0.0,
    class_whitelist: Iterable[str] | None = None,
    accepted_statuses: Iterable[str] | None = ("accepted",),
    box_keys: Iterable[str] = _DEFAULT_BOX_KEYS,
    load_annotations: bool = True,
) -> tuple[BBoxFrameRecord, ...]:
    root = resolve_vivid_annotation_root(
        annotation_root,
        require_annotations=load_annotations,
    )
    modality = str(modality).lower()
    if modality not in {"thermal", "rgb"}:
        raise ValueError("VIVID++ modality must be thermal or rgb")
    if view_mode not in {"single", "multiview"}:
        raise ValueError("VIVID++ view_mode must be single or multiview")
    if view_mode == "multiview" and modality != "thermal":
        raise ValueError("VIVID++ multiview uses thermal as the target modality")
    whitelist = None if class_whitelist is None else {str(value) for value in class_whitelist}
    statuses = None if accepted_statuses is None else {str(value) for value in accepted_statuses}
    selected_box_keys = tuple(str(key) for key in box_keys)
    frames_path = root / "frames.jsonl"
    detections_path = root / "detections.jsonl"
    def file_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return stat.st_size, stat.st_mtime_ns
    cache_key = repr(
        (
            "vivid-annotation-v2",
            str(root),
            file_signature(frames_path),
            file_signature(detections_path),
            modality,
            view_mode,
            float(min_score),
            None if whitelist is None else tuple(sorted(whitelist)),
            None if statuses is None else tuple(sorted(statuses)),
            selected_box_keys,
            bool(load_annotations),
        )
    )
    cache_path = _default_annotation_cache(root, cache_key)
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("key") == cache_key and all(
            view.path.is_file()
            for record in payload["records"]
            for view in record.views
        ):
            return tuple(payload["records"])
    except (OSError, EOFError, KeyError, AttributeError, pickle.UnpicklingError):
        pass

    detections: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    if load_annotations:
        with (root / "detections.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    detections[_frame_key(record)].append(record)

    records: list[BBoxFrameRecord] = []
    with (root / "frames.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            frame = json.loads(line)
            status = frame.get("status")
            if statuses is not None and status not in statuses:
                continue
            thermal_value = frame.get("thermal_path")
            if thermal_value is None:
                raise ValueError(f"VIVID++ frame has no thermal_path: {root}/frames.jsonl:{line_number}")
            thermal_path = _resolve_image(root, str(thermal_value))
            rgb_value = frame.get("rgb_path")
            rgb_path = None if rgb_value is None else _resolve_image(root, str(rgb_value))
            thermal_boxes: list[BBoxAnnotation] = []
            rgb_boxes: list[BBoxAnnotation] = []
            for detection in detections.get(_frame_key(frame), ()):
                thermal_box = _box_from_record(detection, selected_box_keys, min_score, whitelist)
                if thermal_box is not None:
                    thermal_boxes.append(thermal_box)
                if view_mode == "multiview" or modality == "rgb":
                    rgb_box = _box_from_record(detection, ("bbox_rgb_xyxy",), min_score, whitelist)
                    if rgb_box is not None:
                        rgb_boxes.append(rgb_box)
            sequence = Path(str(frame.get("bag", "vivid_pp"))).stem
            frame_id = str(frame.get("thermal_sequence", frame.get("rgb_sequence", line_number)))
            thermal_view = BBoxViewRecord(
                "thermal",
                thermal_path,
                tuple(thermal_boxes),
                annotations_available=load_annotations,
            )
            views = (thermal_view,)
            if view_mode == "multiview":
                if rgb_path is None:
                    raise ValueError(f"VIVID++ multiview frame has no rgb_path: {root}/frames.jsonl:{line_number}")
                views = (
                    BBoxViewRecord(
                        "rgb",
                        rgb_path,
                        tuple(rgb_boxes),
                        annotations_available=load_annotations,
                    ),
                    thermal_view,
                )
            elif modality == "rgb":
                if rgb_path is None:
                    raise ValueError(f"VIVID++ RGB frame has no rgb_path: {root}/frames.jsonl:{line_number}")
                views = (
                    BBoxViewRecord(
                        "rgb",
                        rgb_path,
                        tuple(rgb_boxes),
                        annotations_available=load_annotations,
                    ),
                )
            records.append(
                BBoxFrameRecord(
                    sequence=sequence,
                    frame_id=frame_id,
                    views=views,
                    timestamp=_timestamp(frame),
                    metadata={
                        "bag": frame.get("bag"),
                        "status": status,
                        "rgb_thermal_delta_ms": frame.get("rgb_thermal_delta_ms"),
                        "lidar_rgb_delta_ms": frame.get("lidar_rgb_delta_ms"),
                        "annotation_root": str(root),
                    },
                )
            )
    records.sort(key=lambda record: (record.sequence, record.timestamp if record.timestamp is not None else 0.0, record.frame_id))
    if not records:
        raise ValueError("VIVID++ selection produced no accepted frame records")
    result = tuple(records)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary_path.open("wb") as handle:
            pickle.dump(
                {"key": cache_key, "records": result},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary_path.replace(cache_path)
    except OSError:
        pass
    return result

# 文件讲解：
# 1. frames.jsonl 描述可用同步帧，detections.jsonl 描述每个候选框；
#    _frame_key 用 bag、thermal sequence/path 把两者精确关联。
# 2. `load_annotations=False` 只读取 frames.jsonl，因此 manifest-only style
#    不需要 detections.jsonl；有框模式再按 score、status、类别和 box_keys 过滤。
# 3. thermal single view 只输出 thermal 框；multiview 输出 RGB, thermal，
#    RGB 框来自 bbox_rgb_xyxy，从而不混淆两个视角的坐标系。
# 4. generated/raw 索引均使用内容 stat 和筛选参数组成的 cache key；时间 clip、
#    16-bit 读取、condition 绘制和 batch collation 均由 shared common.py 负责。
