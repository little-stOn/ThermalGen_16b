from __future__ import annotations

from pathlib import Path
from inspect import signature

import pytest

from dwm.common import (
    ConfigError,
    ReadonlyDictIndices,
    SerializedReadonlyDict,
    SerializedReadonlyList,
    TASK_CONFIG_NAMES,
    global_state,
    instantiate_config,
    load_config,
    load_object_from_config,
)

from dwm.datasets.butiv import MotionDataset as ButivMotionDataset
from dwm.datasets.common import BBoxMotionDataset
from dwm.datasets.flir import MotionDataset as FlirMotionDataset
from dwm.datasets.ltir_v1 import MotionDataset as LtirMotionDataset
from dwm.datasets.ms2 import MotionDataset as Ms2MotionDataset
from dwm.datasets.vivid_pp import MotionDataset as VividMotionDataset
from dwm.datasets.zut_fir_adas import MotionDataset as ZutMotionDataset
from dwm.datasets.lynred_mobility import MotionDataset as LynredMotionDataset
from dwm.datasets.tartanrgbt import MotionDataset as TartanMotionDataset


def test_recursive_factory_instantiates_nested_dataset_helpers() -> None:
    config = {
        "_class_name": "dwm.datasets.common.ConcatMotionDataset",
        "datasets": [
            {
                "_class_name": "dwm.common.SerializedReadonlyList",
                "items": ["a", "b"],
            }
        ],
        "ratios": [1.0],
    }

    dataset = instantiate_config(config)

    assert len(dataset) == 2
    assert dataset[0] == "a"
    assert dataset[-1] == "b"


def test_config_loader_expands_environment_and_applies_typed_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text(
        "global_state:\n"
        "  root: ${DWM_TEST_ROOT}\n"
        "value:\n"
        "  _class_name: dwm.common.get_state\n"
        "  key: root\n"
        "settings:\n"
        "  batch_size: 1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DWM_TEST_ROOT", "/mounted/bu_tiv")

    loaded = load_config(config_path, ["settings.batch_size=4", "settings.shuffle=true"])
    value, _ = load_object_from_config(config_path, "value", ["settings.batch_size=4"])

    assert value == "/mounted/bu_tiv"
    assert loaded["settings"] == {"batch_size": 4, "shuffle": True}


def test_unresolved_environment_variables_fail_fast(tmp_path: Path) -> None:
    config_path = tmp_path / "unresolved.yaml"
    config_path.write_text("root: ${MISSING_DWM_TEST_ROOT}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unresolved environment variable"):
        load_config(config_path)


def test_readonly_serialized_containers_preserve_lookup_contract() -> None:
    indices = ReadonlyDictIndices(["z", "a", "a"])
    serialized_list = SerializedReadonlyList([{"value": 1}, {"value": 2}])
    serialized_dict = SerializedReadonlyDict({"z": 3, "a": 4})

    assert indices["a"] == 1
    assert indices.get_all_indices("a") == [1, 2]
    assert serialized_list[-1] == {"value": 2}
    assert serialized_dict.keys() == ["a", "z"]
    assert serialized_dict["a"] == 4
    with pytest.raises(KeyError):
        _ = serialized_dict["missing"]


def test_global_state_is_reinitialized_from_each_config(tmp_path: Path) -> None:
    global_state.clear()
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("global_state:\n  token: first\nvalue: 1\n", encoding="utf-8")
    second.write_text("global_state:\n  token: second\nvalue: 2\n", encoding="utf-8")

    load_object_from_config(first, "value")
    assert global_state["token"] == "first"
    load_object_from_config(second, "value")
    assert global_state["token"] == "second"

def test_butiv_config_declares_dataset_and_loader_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("BUTIV_ROOT", "/mounted/bu_tiv")

    config = load_config(project_root / "configs/datasets/butiv.yaml")

    dataset = config["dataset"]
    assert dataset["_class_name"] == "dwm.datasets.common.DatasetAdapter"
    assert dataset["base_dataset"]["_class_name"] == "dwm.datasets.butiv.MotionDataset"
    assert dataset["base_dataset"]["dataset_root"]["_class_name"] == "dwm.common.get_state"
    assert config["dataloader"]["_class_name"] == "torch.utils.data.DataLoader"
    assert config["dataloader"]["collate_fn"]["_class_name"] == (
        "dwm.datasets.common.CollateFnIgnoring"
    )


def test_butiv_mix_config_declares_official_mix_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("BUTIV_ROOT", "/mounted/bu_tiv")

    config = load_config(project_root / "configs/datasets/butiv_mix.yaml")
    mix_config = config["mix_config"]

    assert set(mix_config) == {"256-256", "192-192", "128-128"}
    assert [value[0] for value in mix_config.values()] == [0.6, 0.3, 0.1]
    assert {option[0] for value in mix_config.values() for option in value[1]} == {4}
    assert {option[1] for value in mix_config.values() for option in value[1]} == {2, 4, 6}
    assert "batch_size" not in config["dataloader"]

def test_all_local_loader_signatures_drop_opendwm_parameters() -> None:
    loader_classes = (
        BBoxMotionDataset,
        ButivMotionDataset,
        FlirMotionDataset,
        LtirMotionDataset,
        ZutMotionDataset,
        LynredMotionDataset,
        TartanMotionDataset,
        Ms2MotionDataset,
        VividMotionDataset,
    )
    removed = {
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
        "render_bbox_images",
        "bbox_image_settings",
        "image_transform",
    }

    for loader_class in loader_classes:
        assert removed.isdisjoint(signature(loader_class).parameters), loader_class.__name__


def test_dataset_configs_only_use_clean_loader_api(monkeypatch: pytest.MonkeyPatch) -> None:
    project_root = Path(__file__).resolve().parents[1]
    for variable in (
        "BUTIV_ROOT",
        "FLIR_ROOT",
        "LTIR_ROOT",
        "ZUT_ROOT",
        "MS2_ROOT",
        "MS2_ANNOTATION_ROOT",
        "VIVID_ROOT",
        "VIVID_ANNOTATION_ROOT",
        "LYNRED_ROOT",
        "TARTAN_ROOT",
    ):
        monkeypatch.setenv(variable, "/configured")
    removed = {
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
        "render_bbox_images",
        "bbox_image_settings",
        "image_transform",
    }

    for name in (
        "butiv",
        "butiv_mix",
        "flir",
        "ltir_v1",
        "zut_fir_adas",
        "ms2",
        "vivid_pp",
        "multi_bbox",
        "multi_style",
        "multi_joint",
    ):
        config = load_config(project_root / "configs/datasets" / f"{name}.yaml")
        keys: set[str] = set()

        def visit(value: object) -> None:
            if isinstance(value, dict):
                keys.update(str(key) for key in value)
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(config)
        assert not removed.intersection(keys), (name, sorted(removed.intersection(keys)))


def test_task_config_names_select_all_three_training_views() -> None:
    assert TASK_CONFIG_NAMES == {
        "bbox": "multi_bbox.yaml",
        "style": "multi_style.yaml",
        "joint": "multi_joint.yaml",
    }
