from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from dwm.datasets.butiv import MotionDataset


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((6, 8), value, dtype=np.uint16)).save(path)


def _write_xml(
    path: Path,
    frames: list[list[dict[str, str]]],
    frame_numbers: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("dataset")
    numbers = list(range(len(frames))) if frame_numbers is None else frame_numbers
    for number, objects in zip(numbers, frames, strict=True):
        frame = ET.SubElement(root, "frame", number=str(number))
        for attributes in objects:
            ET.SubElement(frame, "object", attributes)
    ET.ElementTree(root).write(path, encoding="utf-8")


def _make_sequence(root: Path, sequence: str, image_dir: str, annotation: str, frame_count: int, empty: int | None = None) -> None:
    for index in range(frame_count):
        _write_png(root / image_dir / f"frame_{index:05d}.png", 50_000 + index)
    frames = []
    for index in range(frame_count):
        if index == empty:
            frames.append([])
        else:
            frames.append(
                [
                    {
                        "category": "car",
                        "id": f"{sequence}-track-7",
                        "x1": "1",
                        "y1": "1",
                        "x2": "6",
                        "y2": "5",
                    }
                ]
            )
    _write_xml(root / annotation, frames)


@pytest.fixture
def butiv_root(tmp_path: Path) -> Path:
    _make_sequence(
        tmp_path,
        "marathon_2",
        "marathon/images/seq2/nuc",
        "marathon/marathon_2_2d.xml",
        8,
        empty=2,
    )
    _make_sequence(
        tmp_path,
        "marathon_3",
        "marathon/images/seq3/nuc",
        "marathon/marathon_3_2d.xml",
        8,
    )
    _make_sequence(
        tmp_path,
        "marathon_4",
        "marathon/images/seq4/nuc",
        "marathon/marathon_4_2d.xml",
        8,
    )
    _make_sequence(
        tmp_path,
        "atrium_orange",
        "atrium/images/orange/nuc",
        "atrium/atrium_orange_2d.xml",
        8,
    )
    _make_sequence(
        tmp_path,
        "atrium_red",
        "atrium/images/red/nuc",
        "atrium/atrium_red_2d.xml",
        8,
    )
    return tmp_path


def test_positive_fps_samples_target_time_and_returns_scalar(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="train",
        sequence_length=2,
        fps_stride_tuples=[(10.0, 0.0)],
    )

    item = dataset[0]

    assert item["fps"].ndim == 0
    assert item["pts"].shape == (2, 1)
    assert item["pts"][0, 0].item() == pytest.approx(0.0)
    assert item["pts"][1, 0].item() == pytest.approx(100.0)


def test_zero_fps_uses_index_stride_but_real_millisecond_pts(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="train",
        sequence_length=2,
        fps_stride_tuples=[(0.0, 2.0)],
    )
    item = dataset[1]

    assert item["fps"].item() == 0.0
    assert item["pts"][1, 0].item() == pytest.approx(1000.0 / 30.0)


def test_empty_frames_are_not_removed_from_temporal_index(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="train",
        sequence_length=2,
        fps_stride_tuples=[(0.0, 1.0)],
        include_empty_frames=False,
    )

    marathon_starts = [
        item["segment"][0][0]
        for item in dataset.items
        if item["scene"] == "marathon_2"
    ]

    assert marathon_starts == [0, 3, 4, 5, 6]


def test_default_batch_contract_omits_variable_annotations_and_stacks_scalar_fps(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="train",
        sequence_length=2,
        fps_stride_tuples=[(0.0, 1.0)],
    )

    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=MotionDataset.collate_fn)))

    assert batch["fps"].shape == (2,)
    assert batch["pts"].shape == (2, 2, 1)
    assert len(batch["images"]) == 2
    assert "boxes" not in batch
    assert "labels" not in batch


def test_native_annotations_and_track_ids_are_available_with_custom_collate(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="train",
        sequence_length=1,
        fps_stride_tuples=[(0.0, 1.0)],
        return_annotations=True,
    )

    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=MotionDataset.collate_fn)))

    assert len(batch["boxes"]) == 2
    assert batch["track_ids"][0][0][0] == ("marathon_2-track-7",)


def test_uint16_images_are_fixed_range_float_pil_images(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="val",
        sequence_length=1,
        fps_stride_tuples=[(0.0, 1.0)],
    )

    image = dataset[0]["images"][0][0]

    assert image.mode == "F"
    assert np.asarray(image)[0, 0] == pytest.approx(50_000.0 / 65_535.0, abs=1e-5)




def test_uint16_full_range_maps_monotonically_without_signed_overflow() -> None:
    values = np.arange(65_536, dtype=np.uint16).reshape(256, 256)
    image = MotionDataset.uint16_to_float_image(Image.fromarray(values))
    restored = np.asarray(image) * 65_535.0

    assert image.mode == "F"
    assert np.all(np.diff(restored.ravel()) >= 0.0)
    assert restored[0, 0] == pytest.approx(0.0)
    assert restored[-1, -1] == pytest.approx(65_535.0, abs=1e-2)


def test_atrium_views_are_returned_as_synchronized_t_by_v_items(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="test",
        sequence_length=2,
        fps_stride_tuples=[(0.0, 1.0)],
        view_mode="multiview",
    )

    item = dataset[0]

    assert dataset.view_names == ["CAM_ORANGE", "CAM_RED"]
    assert [len(row) for row in item["images"]] == [2, 2]


def test_default_mode_keeps_view_count_fixed_across_mixed_scenes(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split=None,
        sequence_length=1,
        fps_stride_tuples=[(0.0, 1.0)],
    )

    assert dataset.view_count == 1
    assert dataset.view_names is None
    assert dataset.views_by_scene["marathon_2"] == ["CAM_FRONT"]


def test_single_view_mode_exposes_each_atrium_view_as_independent_samples(
    butiv_root: Path,
) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="test",
        sequence_length=1,
        fps_stride_tuples=[(0.0, 1.0)],
    )

    item_views = [tuple(item["views"]) for item in dataset.items]

    assert dataset.view_count == 1
    assert dataset.view_names is None
    assert item_views.count(("CAM_ORANGE",)) == 8
    assert item_views.count(("CAM_RED",)) == 8
    assert dataset.views_by_scene["atrium"] == ["CAM_ORANGE", "CAM_RED"]


def test_single_view_mode_honors_explicit_atrium_view(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="test",
        sensor_channels=["CAM_RED"],
        sequence_length=1,
        fps_stride_tuples=[(0.0, 1.0)],
    )

    assert dataset.view_names == ["CAM_RED"]
    assert all(item["views"] == ["CAM_RED"] for item in dataset.items)


def test_multiview_mode_rejects_single_view_scenes(butiv_root: Path) -> None:
    with pytest.raises(ValueError, match="at least two views per scene"):
        MotionDataset(
            dataset_root=butiv_root,
            split="train",
            view_mode="multiview",
        )
    with pytest.raises(ValueError, match="at least two sensor channels"):
        MotionDataset(
            dataset_root=butiv_root,
            split="test",
            view_mode="multiview",
            sensor_channels=["CAM_ORANGE"],
        )


def test_tensor_fields_are_stacked_after_adapter_and_shape_errors_are_explicit() -> None:
    def item(fill: float, condition_shape: tuple[int, ...] = (1, 2, 2)) -> dict:
        return {
            "fps": torch.tensor(10.0),
            "pts": torch.zeros((1, 1)),
            "images": [[torch.full(condition_shape, fill)]],
            "bbox_condition_images": [[torch.full(condition_shape, fill)]],
            "sequence": "scene",
            "sample_ids": [["sample"]],
        }

    batch = MotionDataset.collate_fn([item(1.0), item(2.0)])

    assert batch["images"].shape == (2, 1, 1, 1, 2, 2)
    assert batch["bbox_condition_images"].shape == (2, 1, 1, 1, 2, 2)
    with pytest.raises(ValueError, match="tensor shapes differ"):
        MotionDataset.collate_fn([item(1.0), item(2.0, (1, 3, 3))])


def test_collate_rejects_mixed_tensor_and_pil_leaves() -> None:
    tensor_item = {
        "fps": torch.tensor(10.0),
        "pts": torch.zeros((1, 1)),
        "images": [[torch.zeros((1, 2, 2))]],
        "sample_ids": [["tensor"]],
    }
    pil_item = {
        "fps": torch.tensor(10.0),
        "pts": torch.zeros((1, 1)),
        "images": [[Image.new("F", (2, 2))]],
        "sample_ids": [["pil"]],
    }

    with pytest.raises(TypeError, match="mixed tensor and non-tensor"):
        MotionDataset.collate_fn([tensor_item, pil_item])


def test_geometry_transform_keeps_boxes_conditions_and_ids_aligned(butiv_root: Path) -> None:
    class HorizontalFlip:
        def __call__(self, image, boxes, labels, track_ids):
            width = image.width
            flipped = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped_boxes = [(width - x2, y1, width - x1, y2) for x1, y1, x2, y2 in boxes]
            return flipped, flipped_boxes, labels, track_ids

    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="val",
        sequence_length=1,
        fps_stride_tuples=[(0.0, 1.0)],
        geometry_transform=HorizontalFlip(),
        return_annotations=True,
        bbox_condition_settings={"pen_width": 1},
    )

    item = dataset[0]

    assert item["boxes"][0][0][0].tolist() == [2.0, 1.0, 7.0, 5.0]
    assert item["track_ids"][0][0] == ("marathon_4-track-7",)


def test_random_geometry_transform_is_sampled_once_per_clip(butiv_root: Path) -> None:
    class SampledFlip:
        def __init__(self) -> None:
            self.sample_calls = 0

        def sample_clip(self):
            self.sample_calls += 1

            def flip(image, boxes, labels, track_ids):
                width = image.width
                transformed_boxes = [
                    (width - x2, y1, width - x1, y2)
                    for x1, y1, x2, y2 in boxes
                ]
                return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), transformed_boxes, labels, track_ids

            return flip

    transform = SampledFlip()
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="val",
        sequence_length=3,
        fps_stride_tuples=[(0.0, 1.0)],
        geometry_transform=transform,
    )

    dataset[0]

    assert transform.sample_calls == 1




def test_invalid_synthetic_sampling_settings_are_rejected(butiv_root: Path) -> None:
    with pytest.raises(ValueError, match="integer index stride"):
        MotionDataset(
            dataset_root=butiv_root,
            split="val",
            fps_stride_tuples=[(0.0, 1.5)],
        )
    with pytest.raises(ValueError, match="exceeds synthetic timebase"):
        MotionDataset(
            dataset_root=butiv_root,
            split="val",
            fps_stride_tuples=[(31.0, 1.0)],
        )
    with pytest.raises(ValueError, match="finite"):
        MotionDataset(
            dataset_root=butiv_root,
            split="val",
            fps_stride_tuples=[(float("nan"), 1.0)],
        )






def test_bbox_condition_name_and_split_metadata(butiv_root: Path) -> None:
    dataset = MotionDataset(
        dataset_root=butiv_root,
        split="val",
        sequence_length=2,
        fps_stride_tuples=[(0.0, 1.0)],
        bbox_condition_settings={"pen_width": 1},
    )

    item = dataset[0]

    assert [len(row) for row in item["bbox_condition_images"]] == [1, 1]
    assert dataset.dataset_metadata["semantics"].startswith("custom sequence-level")
    assert dataset.dataset_metadata["timebase_fps"] == 30.0
    assert dataset.dataset_metadata["source_fps_verified"] is False
    assert MotionDataset._color_for_label(
        "human.pedestrian", {}
    ) != MotionDataset._color_for_label("vehicle.car", {})




def test_clip_cannot_cross_an_observable_frame_gap(tmp_path: Path) -> None:
    image_dir = tmp_path / "marathon/images/seq4/nuc"
    for frame_number in (0, 1, 3, 4):
        _write_png(image_dir / f"frame_{frame_number:05d}.png", 50_000)
    _write_xml(
        tmp_path / "marathon/marathon_4_2d.xml",
        [[] for _ in (0, 1, 3, 4)],
        frame_numbers=[0, 1, 3, 4],
    )

    with pytest.raises(ValueError, match="no clips"):
        MotionDataset(
            dataset_root=tmp_path,
            split="val",
            sequence_length=3,
            fps_stride_tuples=[(0.0, 1.0)],
        )

@pytest.mark.parametrize(
    "deprecated_parameter",
    [
        "fs",
        "dataset_name",
        "keyframe_only",
        "enable_synchronization_check",
        "enable_scene_description",
        "enable_camera_transforms",
        "enable_ego_transforms",
        "enable_sample_data",
        "_3dbox_image_settings",
        "hdmap_image_settings",
        "image_segmentation_settings",
        "foreground_region_image_settings",
        "_3dbox_bev_settings",
        "hdmap_bev_settings",
        "image_description_settings",
        "stub_key_data_dict",
        "return_native_annotations",
        "image_transform",
    ],
)
def test_removed_compatibility_parameters_are_not_accepted(
    butiv_root: Path, deprecated_parameter: str
) -> None:
    with pytest.raises(TypeError):
        MotionDataset(
            dataset_root=butiv_root,
            split="val",
            **{deprecated_parameter: None},
        )
