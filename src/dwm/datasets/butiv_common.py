"""BU-TIV indexing, synthetic timing, and annotation parsing helpers.

BU-TIV stores 16-bit PNG sequences beside XML annotations. The release does
not provide trustworthy per-frame timestamps through this adapter, so timing
is explicitly a monotonic ordinal timebase. XML frame-number gaps still split
sequences to prevent clips from crossing an observable discontinuity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


DEFAULT_SOURCE_FPS = 30.0
# 仅用于按帧序号构造单调时间轴，不代表真实采集帧率。
TIMESTAMP_SOURCE = "synthetic_ordinal"
DEFAULT_CLASS_NAMES = (
    "human.pedestrian",
    "vehicle.bicycle",
    "vehicle.motorcycle",
    "vehicle.car",
)

_CATEGORY_ALIASES = {
    "people": "human.pedestrian",
    "person": "human.pedestrian",
    "pedestrian": "human.pedestrian",
    "cyclist": "vehicle.bicycle",
    "bicycle": "vehicle.bicycle",
    "motorcyclist": "vehicle.motorcycle",
    "motorcycle": "vehicle.motorcycle",
    "car": "vehicle.car",
}

_FRAME_RE = re.compile(r"^frame_(\d+)\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class SequenceSpec:
    """Static layout and synthetic timebase information for one BU-TIV view."""

    name: str
    image_dir: str
    annotation: str
    frame_mapping: str = "number"
    default_category: str | None = None
    scene: str | None = None
    view: str = "CAM_FRONT"
    source_fps: float = DEFAULT_SOURCE_FPS


@dataclass(frozen=True)
class FrameRecord:
    """One image with validated boxes and explicit synthetic-time metadata."""

    sequence: str
    ordinal: int
    frame_number: int
    image_path: Path
    boxes: tuple[tuple[float, float, float, float], ...]
    labels: tuple[str, ...]
    track_ids: tuple[str | None, ...]
    timestamp_ms: float
    timestamp_source: str
    segment_id: int
    scene: str
    view: str

    @property
    def sample_id(self) -> str:
        return f"butiv:{self.scene}:{self.ordinal:06d}:{self.view}"


# Atrium 的两个目录属于同一场景，但代表两个独立视角；由于 PNG 文件名稀疏，
# 这里按 XML 帧顺序进行对应，而不是直接按文件名匹配。
SEQUENCE_SPECS = {
    "atrium_orange": SequenceSpec(
        name="atrium_orange",
        image_dir="atrium/images/orange/nuc",
        annotation="atrium/atrium_orange_2d.xml",
        frame_mapping="ordinal",
        default_category="human.pedestrian",
        scene="atrium",
        view="CAM_ORANGE",
    ),
    "atrium_red": SequenceSpec(
        name="atrium_red",
        image_dir="atrium/images/red/nuc",
        annotation="atrium/atrium_red_2d.xml",
        frame_mapping="ordinal",
        default_category="human.pedestrian",
        scene="atrium",
        view="CAM_RED",
    ),
    "marathon_2": SequenceSpec(
        name="marathon_2",
        image_dir="marathon/images/seq2/nuc",
        annotation="marathon/marathon_2_2d.xml",
        scene="marathon_2",
    ),
    "marathon_3": SequenceSpec(
        name="marathon_3",
        image_dir="marathon/images/seq3/nuc",
        annotation="marathon/marathon_3_2d.xml",
        scene="marathon_3",
    ),
    "marathon_4": SequenceSpec(
        name="marathon_4",
        image_dir="marathon/images/seq4/nuc",
        annotation="marathon/marathon_4_2d.xml",
        scene="marathon_4",
    ),
}

SPLITS = {
    "train": ("marathon_2", "marathon_3"),
    "val": ("marathon_4",),
    "test": ("atrium_orange", "atrium_red"),
}

SPLIT_METADATA = {
    "train": {
        "sequences": ["marathon_2", "marathon_3"],
        "semantics": "custom sequence-level training split; not the official BU-TIV split",
        "domain_note": "Marathon outdoor sequences",
    },
    "val": {
        "sequences": ["marathon_4"],
        "semantics": "custom sequence-level validation split; not the official BU-TIV split",
        "domain_note": "Marathon outdoor held-out sequence",
    },
    "test": {
        "sequences": ["atrium_orange", "atrium_red"],
        "semantics": "custom cross-scene test split; orange/red are synchronized views of one Atrium scene",
        "domain_note": "Atrium indoor two-view sequence; views are opt-in",
    },
    "all": {
        "sequences": list(SEQUENCE_SPECS),
        "semantics": "all supported annotated sequences; sequence boundaries remain explicit",
        "domain_note": "mixed Marathon and Atrium domains",
    },
}


def canonical_category(value: str) -> str:
    """Normalize a BU-TIV XML category to the shared label vocabulary."""

    normalized = value.strip().lower().replace(" ", "_")
    return _CATEGORY_ALIASES.get(normalized, normalized)


def _parse_float(value: str, field: str, source: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}={value!r} in {source}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field} in {source}")
    return result


def _parse_box(
    object_node: ET.Element, spec: SequenceSpec, source: Path
) -> tuple[tuple[float, ...], str, str | None]:
    attributes = object_node.attrib
    box_keys = ("x1", "y1", "x2", "y2")
    if not all(key in attributes for key in box_keys):
        raise ValueError(
            f"BU-TIV sequence {spec.name} contains a non-rectangular object "
            f"in {source}; point annotations are not converted to boxes"
        )
    x1 = _parse_float(attributes["x1"], "x1", source)
    y1 = _parse_float(attributes["y1"], "y1", source)
    x2 = _parse_float(attributes["x2"], "x2", source)
    y2 = _parse_float(attributes["y2"], "y2", source)
    if not x1 < x2 or not y1 < y2:
        raise ValueError(f"Degenerate box in {source}: {(x1, y1, x2, y2)}")
    category = attributes.get("category") or spec.default_category
    if not category:
        raise ValueError(f"Missing category for BU-TIV box in {source}")
    track_id = attributes.get("id") or attributes.get("track_id")
    return (x1, y1, x2, y2), canonical_category(category), track_id


def _parse_xml(
    source: Path, spec: SequenceSpec
) -> list[
    tuple[
        int,
        tuple[tuple[float, ...], ...],
        tuple[str, ...],
        tuple[str | None, ...],
    ]
]:
    root = ET.parse(source).getroot()
    result = []
    for frame_node in root.findall(".//frame"):
        number_text = frame_node.attrib.get("number")
        if number_text is None:
            raise ValueError(f"Missing frame number in {source}")
        try:
            frame_number = int(number_text)
        except ValueError as exc:
            raise ValueError(f"Invalid frame number {number_text!r} in {source}") from exc
        boxes: list[tuple[float, ...]] = []
        labels: list[str] = []
        track_ids: list[str | None] = []
        for object_node in frame_node.iter("object"):
            box, label, track_id = _parse_box(object_node, spec, source)
            boxes.append(box)
            labels.append(label)
            track_ids.append(track_id)
        result.append((frame_number, tuple(boxes), tuple(labels), tuple(track_ids)))
    frame_numbers = [item[0] for item in result]
    if any(current <= previous for previous, current in zip(frame_numbers, frame_numbers[1:])):
        raise ValueError(f"Frame numbers must be strictly increasing in {source}")
    if not result:
        raise ValueError(f"No annotated frames found in {source}")
    return result


def _image_number(path: Path) -> int:
    match = _FRAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected BU-TIV image filename: {path}")
    return int(match.group(1))


def _resolve_images(root: Path, spec: SequenceSpec, frame_numbers: list[int]) -> list[Path]:
    image_dir = root / spec.image_dir
    images = sorted(image_dir.glob("*.png"), key=_image_number)
    if len(images) != len(frame_numbers):
        raise ValueError(
            f"BU-TIV {spec.name} has {len(frame_numbers)} XML frames but "
            f"{len(images)} PNG images in {image_dir}"
        )
    # Marathon 文件名中的数字可直接对应 XML 帧号；Atrium 文件名稀疏，
    # 因而按 XML 中的 ordinal 顺序对应图像。
    if spec.frame_mapping == "ordinal":
        return images
    by_number = {_image_number(path): path for path in images}
    missing = [number for number in frame_numbers if number not in by_number]
    if missing:
        raise ValueError(f"Missing BU-TIV image frames in {image_dir}: {missing[:5]}")
    return [by_number[number] for number in frame_numbers]

def _resolve_unannotated_images(root: Path, spec: SequenceSpec) -> tuple[list[int], list[Path]]:
    image_dir = root / spec.image_dir
    images = sorted(image_dir.glob("*.png"), key=_image_number)
    if not images:
        raise ValueError(f"BU-TIV sequence {spec.name} has no PNG images in {image_dir}")
    if spec.frame_mapping == "ordinal":
        frame_numbers = list(range(len(images)))
    else:
        frame_numbers = [_image_number(path) for path in images]
    return frame_numbers, images


def load_sequence_records(
    dataset_root: str | Path,
    sequence: str,
    source_fps: float | None = None,
    load_annotations: bool = True,
    max_frames_per_sequence: int | None = None,
) -> list[FrameRecord]:
    """Load one BU-TIV view with a declared synthetic ordinal timebase."""

    root = Path(dataset_root).expanduser().resolve()
    try:
        spec = SEQUENCE_SPECS[sequence]
    except KeyError as exc:
        raise ValueError(f"Unsupported BU-TIV sequence: {sequence}") from exc
    if source_fps is not None and source_fps <= 0:
        raise ValueError("source_fps must be positive")
    if max_frames_per_sequence is not None and max_frames_per_sequence < 1:
        raise ValueError("max_frames_per_sequence must be positive")
    if load_annotations:
        annotation_path = root / spec.annotation
        frame_data = _parse_xml(annotation_path, spec)
        frame_numbers = [item[0] for item in frame_data]
        image_paths = _resolve_images(root, spec, frame_numbers)
    else:
        frame_numbers, image_paths = _resolve_unannotated_images(root, spec)
        frame_data = [
            (frame_number, (), (), ())
            for frame_number in frame_numbers
        ]
    if max_frames_per_sequence is not None:
        frame_data = frame_data[:max_frames_per_sequence]
        image_paths = image_paths[:max_frames_per_sequence]
    timebase_fps = float(source_fps if source_fps is not None else spec.source_fps)
    records: list[FrameRecord] = []
    segment_id = 0
    previous_frame_number: int | None = None
    for ordinal, ((frame_number, boxes, labels, track_ids), image_path) in enumerate(
        zip(frame_data, image_paths, strict=True)
    ):
        if previous_frame_number is not None and frame_number != previous_frame_number + 1:
            segment_id += 1
        records.append(
            FrameRecord(
                sequence=spec.name,
                ordinal=ordinal,
                frame_number=frame_number,
                image_path=image_path,
                boxes=boxes,
                labels=labels,
                track_ids=track_ids,
                timestamp_ms=ordinal * 1000.0 / timebase_fps,
                timestamp_source=TIMESTAMP_SOURCE,
                segment_id=segment_id,
                scene=spec.scene or spec.name,
                view=spec.view,
            )
        )
        previous_frame_number = frame_number
    return records


def sequence_names(split: str | None) -> tuple[str, ...]:
    if split is None or split == "all":
        return tuple(SEQUENCE_SPECS)
    try:
        return SPLITS[split]
    except KeyError as exc:
        raise ValueError(f"Unknown BU-TIV split: {split}") from exc


def split_metadata(split: str | None) -> dict[str, object]:
    key = "all" if split is None else split
    try:
        metadata = SPLIT_METADATA[key]
    except KeyError as exc:
        raise ValueError(f"Unknown BU-TIV split: {split}") from exc
    return dict(metadata)


def load_records(
    dataset_root: str | Path,
    split: str | None = None,
    source_fps: float | None = None,
    load_annotations: bool = True,
    max_frames_per_sequence: int | None = None,
) -> dict[str, list[FrameRecord]]:
    """Return records grouped by BU-TIV sequence."""

    return {
        name: load_sequence_records(
            dataset_root,
            name,
            source_fps=source_fps,
            load_annotations=load_annotations,
            max_frames_per_sequence=max_frames_per_sequence,
        )
        for name in sequence_names(split)
    }


def load_multiview_records(
    dataset_root: str | Path,
    split: str | None = None,
    source_fps: float | None = None,
    load_annotations: bool = True,
    max_frames_per_sequence: int | None = None,
) -> dict[str, dict[str, list[FrameRecord]]]:
    """Return scene groups with original view records kept separately."""

    scenes: dict[str, dict[str, list[FrameRecord]]] = {}
    for name in sequence_names(split):
        records = load_sequence_records(
            dataset_root,
            name,
            source_fps=source_fps,
            load_annotations=load_annotations,
            max_frames_per_sequence=max_frames_per_sequence,
        )
        if not records:
            continue
        scene = records[0].scene
        views = scenes.setdefault(scene, {})
        if records[0].view in views:
            raise ValueError(f"Duplicate BU-TIV view {records[0].view} in scene {scene}")
        views[records[0].view] = records
    for scene, views in scenes.items():
        lengths = {len(records) for records in views.values()}
        if len(lengths) != 1:
            raise ValueError(f"BU-TIV scene {scene} has unsynchronized view lengths: {lengths}")
        reference_view, reference_records = next(iter(views.items()))
        for view, records in views.items():
            if view == reference_view:
                continue
            mismatched = [
                index
                for index, (reference, candidate) in enumerate(
                    zip(reference_records, records, strict=True)
                )
                if (
                    reference.ordinal != candidate.ordinal
                    or reference.frame_number != candidate.frame_number
                    or reference.segment_id != candidate.segment_id
                    or reference.timestamp_ms != candidate.timestamp_ms
                )
            ]
            if mismatched:
                raise ValueError(
                    f"BU-TIV scene {scene} views {reference_view}/{view} are "
                    f"not ordinal-aligned at frame {mismatched[0]}"
                )
    return scenes



# 文件讲解：
# 1. 本文件是 BU-TIV 的格式解析层，不负责 DataLoader batch 和图像 tensor；
#    它把不同子集的目录、XML 文件和视角规则统一成 FrameRecord。
# 2. load_sequence_records 解析 XML object 的 x1/y1/x2/y2，仅保留真正的
#    矩形 bbox；只有点标注或缺少 XML 的子集不会被伪造成框。
# 3. 每个 FrameRecord 带有 scene、view、frame_number、segment_id 和合成
#    ordinal timestamp。frame gap 会开启新 segment，避免 clip 跨越不连续帧。
# 4. load_multiview_records 按 scene 聚合视角，并校验各视角的 ordinal、
#    frame number、segment 和 timestamp 一致；不一致时立即报错，而不是
#    静默错配 RGB/thermal 画面。
# 5. 维护时先检查这里的路径映射和 split_metadata，再修改 butiv.py；
#    上层 loader 的职责只是采样 clip、读取图像、变换 bbox 和输出字段。
