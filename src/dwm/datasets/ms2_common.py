"""MS2 generated thermal-bbox manifest parsing."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Iterable, Sequence

from dwm.datasets.common import (
    BBoxFrameRecord,
    BBoxViewRecord,
    limit_sequence,
    make_bbox_annotation,
)


def resolve_ms2_project_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "ms2" and root.parent.name == "raw":
        return root.parents[2]
    if (root / "data" / "raw" / "ms2").is_dir():
        return root
    raise FileNotFoundError(f"MS2 raw root does not exist: {root}")


def _resolve_path(value: str, project_root: Path, annotation_root: Path) -> Path:
    raw = Path(str(value))
    candidates = [
        raw if raw.is_absolute() else project_root / raw,
        raw if raw.is_absolute() else annotation_root / raw,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"MS2 manifest image does not exist: {value}")


def _timestamp_seconds(record: dict) -> float | None:
    for key in ("timestamp_thr", "timestamp", "timestamp_ns"):
        value = record.get(key)
        if value is None:
            continue
        value = float(value)
        if abs(value) > 1.0e14:
            return value / 1.0e9
        if abs(value) > 1.0e11:
            return value / 1.0e6
        if abs(value) > 1.0e8:
            return value
        return value
    return None

def _default_index_cache(annotation_root: Path, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return annotation_root / ".dwm_cache" / f"ms2_{digest}.pkl"

def _resolve_ms2_raw_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if root.name == "ms2" and root.parent.name == "raw":
        return root
    project_root = resolve_ms2_project_root(root)
    raw_root = project_root / "data" / "raw" / "ms2"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"MS2 raw root does not exist: {raw_root}")
    return raw_root


def load_ms2_raw_records(
    dataset_root: str | Path,
    sequences: Iterable[str] | None = None,
    thermal_side: str = "img_left",
    max_frames_per_sequence: int | None = None,
    source_fps: float = 10.0,
) -> tuple[BBoxFrameRecord, ...]:
    """Index raw thermal frames without requiring generated annotations."""

    root = _resolve_ms2_raw_root(dataset_root)
    source_fps = float(source_fps)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError("source_fps must be finite and positive")
    sync_root = root / "sync_data"
    if not sync_root.is_dir():
        raise FileNotFoundError(f"MS2 sync root does not exist: {sync_root}")
    selected = None if sequences is None else {str(value) for value in sequences}
    records: list[BBoxFrameRecord] = []
    for sequence_root in sorted(path for path in sync_root.iterdir() if path.is_dir()):
        sequence = sequence_root.name
        if selected is not None and sequence not in selected:
            continue
        thermal_root = sequence_root / "thr" / thermal_side
        if not thermal_root.is_dir():
            continue
        images = sorted(
            [
                path
                for path in thermal_root.iterdir()
                if path.is_file() and path.suffix.lower() == ".png"
            ],
            key=lambda path: (int(path.stem) if path.stem.isdigit() else 0, path.name),
        )
        if not images:
            continue
        timestamp_path = sequence_root / "thr" / f"{thermal_side}_timestamp.txt"
        timestamps = (
            [
                float(line.strip())
                for line in timestamp_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                if line.strip()
            ]
            if timestamp_path.is_file()
            else []
        )
        if timestamps and len(timestamps) != len(images):
            raise ValueError(
                f"MS2 thermal timestamp/image count mismatch for {sequence}: "
                f"{len(timestamps)} != {len(images)}"
            )
        timestamp_values = timestamps or [None] * len(images)
        selected_frames = limit_sequence(
            tuple(enumerate(zip(images, timestamp_values, strict=True))),
            max_frames_per_sequence,
        )
        for index, (image_path, raw_timestamp) in selected_frames:
            timestamp = (
                _timestamp_seconds({"timestamp": raw_timestamp})
                if raw_timestamp is not None
                else index / source_fps
            )
            records.append(
                BBoxFrameRecord(
                    sequence=sequence,
                    frame_id=image_path.stem,
                    timestamp=timestamp,
                    views=(
                        BBoxViewRecord(
                            "thermal",
                            image_path,
                            (),
                            annotations_available=False,
                        ),
                    ),
                    metadata={
                        "raw_root": str(root),
                        "thermal_side": thermal_side,
                        "frame_index": index,
                    },
                )
            )
    if not records:
        raise ValueError("MS2 raw selection produced no thermal frame records")
    return tuple(records)


def merge_ms2_annotations(
    raw_records: Sequence[BBoxFrameRecord],
    annotated_records: Sequence[BBoxFrameRecord],
) -> tuple[BBoxFrameRecord, ...]:
    """Overlay generated boxes onto the raw frame index without duplicating images."""

    annotations = {
        (record.sequence, record.frame_id): record.views[0]
        for record in annotated_records
    }
    merged: list[BBoxFrameRecord] = []
    for record in raw_records:
        annotated_view = annotations.get((record.sequence, record.frame_id))
        if annotated_view is None:
            merged.append(record)
            continue
        merged.append(
            BBoxFrameRecord(
                sequence=record.sequence,
                frame_id=record.frame_id,
                timestamp=record.timestamp,
                views=(
                    BBoxViewRecord(
                        record.views[0].name,
                        record.views[0].path,
                        annotated_view.boxes,
                        annotations_available=True,
                    ),
                ),
                metadata={**(record.metadata or {}), "annotation_source": "generated"},
            )
        )
    return tuple(merged)


def load_ms2_records(
    annotation_root: str | Path,
    dataset_root: str | Path,
    records_file: str | Path = "records.jsonl",
    sequences: Iterable[str] | None = None,
    min_score: float = 0.0,
    class_whitelist: Iterable[str] | None = None,
    accepted_statuses: Iterable[str] | None = None,
    index_cache: str | Path | None = None,
    use_index_cache: bool = True,
) -> tuple[BBoxFrameRecord, ...]:
    annotation_root = Path(annotation_root).expanduser().resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"MS2 annotation root does not exist: {annotation_root}")
    project_root = resolve_ms2_project_root(dataset_root)
    path = Path(records_file)
    if not path.is_absolute():
        path = annotation_root / path
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MS2 records file does not exist: {path}")
    if not 0.0 <= float(min_score) <= 1.0:
        raise ValueError("MS2 min_score must be within [0, 1]")
    selected_sequences = (
        None if sequences is None else tuple(sorted(str(value) for value in sequences))
    )
    statuses = (
        None if accepted_statuses is None else tuple(sorted(str(value) for value in accepted_statuses))
    )
    whitelist = (
        None if class_whitelist is None else tuple(sorted(str(value) for value in class_whitelist))
    )
    cache_key = repr(
        (
            "ms2-index-v1",
            str(path),
            path.stat().st_mtime_ns,
            path.stat().st_size,
            str(project_root),
            selected_sequences,
            float(min_score),
            whitelist,
            statuses,
        )
    )
    cache_path = (
        _default_index_cache(annotation_root, cache_key)
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

    sequences_set = None if selected_sequences is None else set(selected_sequences)
    statuses_set = None if statuses is None else set(statuses)
    whitelist_set = None if whitelist is None else set(whitelist)
    records: list[BBoxFrameRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sequence = str(record.get("sequence", ""))
            if not sequence or (sequences_set is not None and sequence not in sequences_set):
                continue
            status = record.get("frame_status", record.get("status"))
            if statuses_set is not None and status not in statuses_set:
                continue
            image_value = record.get("image") or record.get("image_path") or record.get("thermal_path")
            if image_value is None:
                raise ValueError(f"MS2 record has no thermal image path: {path}:{line_number}")
            image_path = _resolve_path(str(image_value), project_root, annotation_root)
            raw_boxes = record.get("boxes") or []
            raw_labels = record.get("labels") or []
            raw_scores = record.get("scores") or []
            raw_track_ids = record.get("track_ids") or []
            if len(raw_labels) not in {0, len(raw_boxes)}:
                raise ValueError(f"MS2 boxes/labels length mismatch: {path}:{line_number}")
            if len(raw_scores) not in {0, len(raw_boxes)}:
                raise ValueError(f"MS2 boxes/scores length mismatch: {path}:{line_number}")
            if len(raw_track_ids) not in {0, len(raw_boxes)}:
                raise ValueError(f"MS2 boxes/track_ids length mismatch: {path}:{line_number}")
            boxes = []
            for index, raw_box in enumerate(raw_boxes):
                score = float(raw_scores[index]) if raw_scores else None
                if score is not None and score < float(min_score):
                    continue
                label = str(raw_labels[index]) if raw_labels else "vehicle.unknown"
                if whitelist_set is not None and label not in whitelist_set:
                    continue
                track_id = (
                    str(raw_track_ids[index])
                    if raw_track_ids
                    else f"{sequence}:{record.get('frame_id', '')}:{index}"
                )
                boxes.append(make_bbox_annotation(raw_box, label, track_id=track_id, score=score))
            frame_id = str(record.get("frame_id", ""))
            records.append(
                BBoxFrameRecord(
                    sequence=sequence,
                    frame_id=frame_id,
                    timestamp=_timestamp_seconds(record),
                    views=(BBoxViewRecord("thermal", image_path, tuple(boxes)),),
                    metadata={
                        "record_id": record.get("id"),
                        "status": status,
                        "annotation_source": record.get("annotation_source"),
                        "confidence_filter": record.get("confidence_filter"),
                        "timestamp_delta_ms": record.get("timestamp_delta_ms"),
                        "image_bits": record.get("image_bits", 16),
                    },
                )
            )
    records.sort(
        key=lambda record: (
            record.sequence,
            record.timestamp if record.timestamp is not None else 0.0,
            record.frame_id,
        )
    )
    if not records:
        raise ValueError("MS2 selection produced no annotated frame records")
    result = tuple(records)
    if use_index_cache:
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
# 1. load_ms2_records 读取生成的 records.jsonl，把 image/image_path/thermal_path
#    解析成真实文件，并将 boxes、labels、scores、track_ids 对齐成统一对象。
# 2. timestamp_thr 优先按纳秒转换为秒；frame_status、score 和 sequence 过滤
#    在索引阶段完成，避免把 rejected 或低置信度检测带入训练。
# 3. MS2 的 boxes 是左 thermal 图像上的绝对 xyxy，不能再次做 RGB 到 thermal
#    投影；投影工作已经在生成 annotation 阶段完成。
# 4. index_cache 的 key 绑定 records 文件的时间和大小以及筛选参数；修改生成
#    annotation 后应删除缓存或让文件 stat 变化，再重新调试。
