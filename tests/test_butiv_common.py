from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import pytest

from dwm.datasets.butiv_common import (
    canonical_category,
    load_multiview_records,
    load_records,
    load_sequence_records,
)


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.full((4, 4), value, dtype=np.uint16))
    image.save(path)


def _write_xml(path: Path, frame_objects: list[tuple[int, list[dict[str, str]]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("dataset")
    for number, objects in frame_objects:
        frame = ET.SubElement(root, "frame", number=str(number))
        for attributes in objects:
            ET.SubElement(frame, "object", attributes)
    ET.ElementTree(root).write(path, encoding="utf-8")


def test_marathon_records_pair_frame_numbers_and_normalize_categories(tmp_path: Path) -> None:
    root = tmp_path
    _write_png(root / "marathon/images/seq2/nuc/frame_00001.png", 10)
    _write_png(root / "marathon/images/seq2/nuc/frame_00002.png", 20)
    _write_xml(
        root / "marathon/marathon_2_2d.xml",
        [
            (1, [{"category": "people", "id": "track-1", "x1": "1", "y1": "2", "x2": "3", "y2": "4"}]),
            (2, [{"category": "cyclist", "x1": "0", "y1": "0", "x2": "2", "y2": "3"}]),
        ],
    )

    records = load_sequence_records(root, "marathon_2")

    assert [record.frame_number for record in records] == [1, 2]
    assert [record.image_path.name for record in records] == ["frame_00001.png", "frame_00002.png"]
    assert records[0].labels == ("human.pedestrian",)
    assert records[1].labels == ("vehicle.bicycle",)
    assert records[0].boxes == ((1.0, 2.0, 3.0, 4.0),)
    assert records[0].track_ids == ("track-1",)
    assert records[0].timestamp_ms == 0.0
    assert records[0].scene == "marathon_2"
    assert records[0].view == "CAM_FRONT"
    assert records[0].sample_id == "butiv:marathon_2:000000:CAM_FRONT"


def test_atrium_uses_ordinal_mapping_for_sparse_filenames(tmp_path: Path) -> None:
    root = tmp_path
    _write_png(root / "atrium/images/orange/nuc/frame_00090.png", 10)
    _write_png(root / "atrium/images/orange/nuc/frame_00120.png", 20)
    _write_xml(
        root / "atrium/atrium_orange_2d.xml",
        [
            (1, [{"x1": "0", "y1": "0", "x2": "1", "y2": "1"}]),
            (2, []),
        ],
    )

    records = load_sequence_records(root, "atrium_orange")

    assert [record.image_path.name for record in records] == ["frame_00090.png", "frame_00120.png"]
    assert records[0].labels == ("human.pedestrian",)
    assert records[1].boxes == ()


def test_observable_frame_number_gaps_start_new_synthetic_segment(tmp_path: Path) -> None:
    root = tmp_path
    _write_png(root / "marathon/images/seq2/nuc/frame_00001.png", 10)
    _write_png(root / "marathon/images/seq2/nuc/frame_00003.png", 20)
    _write_xml(
        root / "marathon/marathon_2_2d.xml",
        [
            (1, []),
            (3, []),
        ],
    )

    records = load_sequence_records(root, "marathon_2")

    assert [record.segment_id for record in records] == [0, 1]
    assert {record.timestamp_source for record in records} == {"synthetic_ordinal"}


def test_non_increasing_xml_frame_numbers_are_rejected(tmp_path: Path) -> None:
    root = tmp_path
    _write_png(root / "marathon/images/seq2/nuc/frame_00001.png", 10)
    _write_png(root / "marathon/images/seq2/nuc/frame_00002.png", 20)
    _write_xml(
        root / "marathon/marathon_2_2d.xml",
        [
            (2, []),
            (1, []),
        ],
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_sequence_records(root, "marathon_2")


def test_point_annotations_are_rejected_in_bbox_sequences(tmp_path: Path) -> None:
    root = tmp_path
    _write_png(root / "marathon/images/seq2/nuc/frame_00001.png", 10)
    _write_xml(
        root / "marathon/marathon_2_2d.xml",
        [(1, [{"x": "2", "y": "3"}])],
    )

    with pytest.raises(ValueError, match="non-rectangular"):
        load_sequence_records(root, "marathon_2")


def test_split_selection_is_sequence_level(tmp_path: Path) -> None:
    for sequence, image_dir, annotation in [
        ("marathon_2", "marathon/images/seq2/nuc", "marathon/marathon_2_2d.xml"),
        ("marathon_3", "marathon/images/seq3/nuc", "marathon/marathon_3_2d.xml"),
    ]:
        _write_png(tmp_path / image_dir / "frame_00001.png", 10)
        _write_xml(tmp_path / annotation, [(1, [])])

    records = load_records(tmp_path, "train")

    assert tuple(records) == ("marathon_2", "marathon_3")
    assert all(len(value) == 1 for value in records.values())


def test_load_multiview_records_groups_synchronized_atrium_views(tmp_path: Path) -> None:
    for view, image_dir, annotation in [
        (
            "orange",
            "atrium/images/orange/nuc",
            "atrium/atrium_orange_2d.xml",
        ),
        (
            "red",
            "atrium/images/red/nuc",
            "atrium/atrium_red_2d.xml",
        ),
    ]:
        _write_png(tmp_path / image_dir / "frame_00001.png", 10)
        _write_png(tmp_path / image_dir / "frame_00002.png", 20)
        _write_xml(
            tmp_path / annotation,
            [
                (1, [{"x1": "0", "y1": "0", "x2": "1", "y2": "1"}]),
                (2, []),
            ],
        )

    grouped = load_multiview_records(tmp_path, "test")

    assert tuple(grouped) == ("atrium",)
    assert tuple(grouped["atrium"]) == ("CAM_ORANGE", "CAM_RED")
    assert grouped["atrium"]["CAM_ORANGE"][1].timestamp_ms == 1000.0 / 30.0


def test_multiview_records_reject_xml_alignment_mismatch(tmp_path: Path) -> None:
    for image_dir in (
        "atrium/images/orange/nuc",
        "atrium/images/red/nuc",
    ):
        _write_png(tmp_path / image_dir / "frame_00001.png", 10)
        _write_png(tmp_path / image_dir / "frame_00002.png", 20)
    _write_xml(
        tmp_path / "atrium/atrium_orange_2d.xml",
        [(1, []), (2, [])],
    )
    _write_xml(
        tmp_path / "atrium/atrium_red_2d.xml",
        [(1, []), (3, [])],
    )

    with pytest.raises(ValueError, match="not ordinal-aligned"):
        load_multiview_records(tmp_path, "test")


def test_unknown_categories_remain_explicit() -> None:
    assert canonical_category("unusual target") == "unusual_target"
