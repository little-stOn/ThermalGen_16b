from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from dwm.datasets.common import (
    BBoxFrameRecord,
    BBoxMotionDataset,
    BBoxViewRecord,
    make_bbox_annotation,
)
from dwm.datasets.flir_common import load_flir_records
from dwm.datasets.ltir_v1_common import load_ltir_records
from dwm.datasets.ms2_common import load_ms2_records
from dwm.datasets.vivid_pp_common import load_vivid_records
from dwm.datasets.zut_fir_adas_common import load_zut_records
from dwm.datasets.lynred_mobility_common import load_lynred_records
from dwm.datasets.tartanrgbt_common import load_tartan_records


def _write_image(path: Path, mode: str = "L", size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)


def _write_coco(path: Path, image_names: list[str], video_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": [
            {
                "id": index,
                "file_name": name,
                "width": 8,
                "height": 6,
                "extra_info": {"video_id": video_id},
            }
            for index, name in enumerate(image_names)
        ],
        "annotations": [
            {"id": index, "image_id": index, "category_id": 3, "bbox": [1, 1, 2, 3]}
            for index in range(len(image_names))
        ],
        "categories": [{"id": 3, "name": "car"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_flir_parser_loads_single_and_paired_video_views(tmp_path: Path) -> None:
    root = tmp_path / "FLIR_ADAS_v2"
    thermal_names = [f"data/video-th-frame-{index:06d}-x.jpg" for index in range(2)]
    for name in thermal_names:
        _write_image(root / "images_thermal_train" / name)
    _write_coco(root / "images_thermal_train/coco.json", thermal_names, "th")

    rgb_name = "data/video-rgb-frame-000000-x.jpg"
    thermal_name = "data/video-th-test-frame-000000-x.jpg"
    _write_image(root / "video_rgb_test" / rgb_name)
    _write_image(root / "video_thermal_test" / thermal_name)
    _write_coco(root / "video_rgb_test/coco.json", [rgb_name], "rgb")
    _write_coco(root / "video_thermal_test/coco.json", [thermal_name], "th-test")
    (root.parent / "rgb_to_thermal_vid_map.json").write_text(
        json.dumps({Path(rgb_name).name: Path(thermal_name).name}), encoding="utf-8"
    )

    single = load_flir_records(root, "train", "thermal", "single")
    paired = load_flir_records(root, "test", "thermal", "multiview")

    assert len(single) == 2
    assert single[0].views[0].boxes[0].label == "vehicle.car"
    assert single[0].views[0].boxes[0].xyxy == (1.0, 1.0, 3.0, 4.0)
    assert len(paired) == 1
    assert [view.name for view in paired[0].views] == ["rgb", "thermal"]


def test_ltir_parser_matches_groundtruth_to_numeric_frames(tmp_path: Path) -> None:
    root = tmp_path / "ltir_v1_0_8bit_16bit" / "16_car"
    _write_image(root / "00000001.png", "L", (640, 480))
    _write_image(root / "00000002.png", "L", (640, 480))
    (root / "groundtruth.txt").write_text(
        "10,20,10,80,70,80,70,20\n11,21,11,81,71,81,71,21\n",
        encoding="utf-8",
    )

    records = load_ltir_records(tmp_path, mode="16bit")

    assert len(records) == 2
    assert records[0].views[0].boxes[0].label == "vehicle.car"
    assert records[0].views[0].boxes[0].xyxy == (10.0, 20.0, 70.0, 80.0)


def test_zut_parser_keeps_temporal_empty_frames_but_parses_yolo_boxes(tmp_path: Path) -> None:
    recording = tmp_path / "zut_fir_adas/Denmark/Copenhagen/run"
    frame0 = recording / "16BitFrames/frameIndex_0.png"
    frame1 = recording / "16BitFrames/frameIndex_1.png"
    _write_image(frame0, "I;16", (320, 240))
    _write_image(frame1, "I;16", (320, 240))
    (recording / "annotations/frameIndex_0.txt").parent.mkdir(parents=True, exist_ok=True)
    (recording / "annotations/frameIndex_0.txt").write_text("0 0.5 0.5 0.25 0.5\n", encoding="utf-8")
    (recording / "annotations/frameIndex_1.txt").write_text("", encoding="utf-8")

    records = load_zut_records(tmp_path, split="all", include_empty_frames=True)

    assert len(records) == 2
    assert records[0].views[0].boxes[0].label == "human.pedestrian"
    assert records[0].views[0].boxes[0].xyxy == (120.0, 60.0, 200.0, 180.0)
    assert records[1].views[0].boxes == ()
    style_records = load_zut_records(
        tmp_path,
        split="all",
        include_empty_frames=False,
        load_annotations=False,
        max_frames_per_sequence=1,
    )
    assert len(style_records) == 1
    assert style_records[0].views[0].annotations_available is False


def test_ms2_parser_resolves_project_relative_generated_manifest(tmp_path: Path) -> None:
    project = tmp_path / "project"
    dataset_root = project / "data/raw/ms2"
    image = dataset_root / "sync_data/seq/thr/img_left/000000.png"
    _write_image(image, "I;16", (640, 256))
    annotation_root = project / "data/standardized/annotations/ms2"
    annotation_root.mkdir(parents=True)
    record = {
        "id": "ms2:seq:000000",
        "sequence": "seq",
        "frame_id": "000000",
        "image": "data/raw/ms2/sync_data/seq/thr/img_left/000000.png",
        "boxes": [[10, 20, 50, 80]],
        "labels": ["vehicle.car"],
        "scores": [0.95],
        "frame_status": "accepted_highconf",
        "timestamp_thr": 1_700_000_000_000_000_000,
    }
    (annotation_root / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    records = load_ms2_records(annotation_root, dataset_root)

    assert len(records) == 1
    assert records[0].views[0].path == image.resolve()
    assert records[0].views[0].boxes[0].score == 0.95
    assert records[0].timestamp == 1_700_000_000.0


def test_vivid_parser_builds_thermal_and_rgb_views_from_generated_records(tmp_path: Path) -> None:
    root = tmp_path / "vivid"
    thermal = "images/shard00/thermal/00000000.png"
    rgb = "images/shard00/rgb/00000000.png"
    _write_image(root / thermal, "I;16", (640, 512))
    _write_image(root / rgb, "RGB", (1280, 1024))
    root.mkdir(parents=True, exist_ok=True)
    frame = {
        "bag": "/bags/campus_day1.bag",
        "thermal_sequence": 0,
        "rgb_sequence": 0,
        "thermal_stamp": 1.0,
        "status": "accepted",
        "thermal_path": thermal,
        "rgb_path": rgb,
    }
    detection = {
        **frame,
        "class_name": "vehicle.car",
        "score": 0.9,
        "bbox_thr_lidar_xyxy": [10, 20, 80, 100],
        "bbox_rgb_xyxy": [20, 40, 160, 200],
    }
    (root / "frames.jsonl").write_text(json.dumps(frame) + "\n", encoding="utf-8")
    (root / "detections.jsonl").write_text(json.dumps(detection) + "\n", encoding="utf-8")

    records = load_vivid_records(root, view_mode="multiview")

    assert len(records) == 1
    assert [view.name for view in records[0].views] == ["rgb", "thermal"]
    assert records[0].views[0].boxes[0].xyxy == (20.0, 40.0, 160.0, 200.0)
    assert records[0].views[1].boxes[0].xyxy == (10.0, 20.0, 80.0, 100.0)
    (root / "detections.jsonl").unlink()
    style_records = load_vivid_records(
        root,
        view_mode="single",
        load_annotations=False,
    )
    assert style_records[0].views[0].boxes == ()
    assert style_records[0].views[0].annotations_available is False


def test_bbox_annotation_rejects_invalid_geometry() -> None:
    try:
        make_bbox_annotation([0, 0, 0, 1], "vehicle.car")
    except ValueError as error:
        assert "positive area" in str(error)
    else:
        raise AssertionError("invalid bbox was accepted")

def test_shared_bbox_dataset_emits_temporal_contract(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    paths = [tmp_path / f"{index:02d}.png" for index in range(2)]
    for path in paths:
        _write_image(path, "I;16", (8, 6))
    box = make_bbox_annotation([1, 1, 4, 4], "vehicle.car", "track-1")
    records = tuple(
        BBoxFrameRecord(
            sequence="scene",
            frame_id=str(index),
            views=(BBoxViewRecord("thermal", path, (box,)),),
        )
        for index, path in enumerate(paths)
    )

    dataset = BBoxMotionDataset(
        records,
        sequence_length=2,
        fps_stride_tuples=[(0.0, 1.0)],
    )
    item = dataset[0]

    assert item["images"][0][0].size == (8, 6)
    assert item["bbox_condition_images"][0][0].mode == "RGB"


def test_annotation_modes_distinguish_style_and_joint_contracts(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    paths = [tmp_path / f"{index:02d}.png" for index in range(2)]
    for path in paths:
        _write_image(path, "I;16", (8, 6))
    box = make_bbox_annotation([1, 1, 4, 4], "vehicle.car", "track-1")
    records = tuple(
        BBoxFrameRecord(
            sequence="scene",
            frame_id=str(index),
            views=(BBoxViewRecord("thermal", path, (box,)),),
        )
        for index, path in enumerate(paths)
    )

    style = BBoxMotionDataset(
        records,
        sequence_length=2,
        annotation_mode="none",
        return_annotations=False,
    )
    joint = BBoxMotionDataset(
        records,
        sequence_length=2,
        annotation_mode="optional",
        return_annotations=True,
    )
    style_item = style[0]
    joint_item = joint[0]

    assert "bbox_condition_images" not in style_item
    assert not style_item["bbox_available"].any()
    assert not style_item["condition_valid"].any()
    assert joint_item["bbox_condition_images"][0][0].mode == "RGB"
    assert joint_item["bbox_available"].all()
    assert joint_item["condition_valid"].all()
    unavailable_records = tuple(
        BBoxFrameRecord(
            sequence="unavailable",
            frame_id=str(index),
            views=(
                BBoxViewRecord(
                    "thermal",
                    path,
                    (box,),
                    annotations_available=False,
                ),
            ),
        )
        for index, path in enumerate(paths)
    )
    unavailable = BBoxMotionDataset(
        unavailable_records,
        sequence_length=2,
        annotation_mode="optional",
        return_annotations=True,
    )
    unavailable_item = unavailable[0]
    assert not unavailable_item["condition_valid"].any()
    assert unavailable_item["boxes"][0][0].shape == (0, 4)
    assert unavailable_item["bbox_condition_images"][0][0].getbbox() is None


def test_lynred_parser_reads_metadata_paths_without_bbox(tmp_path: Path) -> None:
    root = tmp_path / "range_dataset"
    relative = Path("summer/rural/day/target/10m")
    sequence_root = root / "16bits" / "qvga" / relative
    for index in range(2):
        _write_image(sequence_root / f"image_{index:08d}.png", "I;16", (8, 6))
    metadata = root / "metadata/metadata.csv"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        "qvga_path;vga_path;environment\n"
        f"qvga/{relative.as_posix()};vga/{relative.as_posix()};rural\n",
        encoding="utf-8",
    )

    records = load_lynred_records(root, resolution="qvga")

    assert len(records) == 2
    assert records[0].views[0].annotations_available is False


def test_tartan_parser_pairs_thermal_stereo_views(tmp_path: Path) -> None:
    root = tmp_path / "tartanrgbt"
    run = root / "day1/run"
    left = run / "thermal_left_rect_16"
    right = run / "thermal_right_rect_16"
    for index in range(2):
        _write_image(left / f"{index:08d}.png", "I;16", (8, 6))
        _write_image(right / f"{index:08d}.png", "I;16", (8, 6))
    (left / "timestamps.txt").write_text("1.0\n1.1\n", encoding="utf-8")

    records = load_tartan_records(root, view_mode="multiview")

    assert len(records) == 2
    assert [view.name for view in records[0].views] == ["thermal_left", "thermal_right"]
    assert records[0].views[0].annotations_available is False
    limited = load_tartan_records(root, view_mode="multiview", max_frames_per_sequence=1)
    assert len(limited) == 1
    assert limited[0].timestamp == pytest.approx(1.1)
