from __future__ import annotations

from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from dwm.common import ConfigError, load_dataloader_from_config
from dwm.utils.sampler import VariableVideoBatchSampler


class _TinyVideoDataset:
    def __init__(self, size: int = 9) -> None:
        self.size = int(size)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int | str) -> dict[str, object]:
        if isinstance(index, str):
            base_index, frames, height, width = (int(value) for value in index.split("-"))
            return {
                "index": base_index,
                "frames": frames,
                "height": height,
                "width": width,
            }
        return {"index": index}


def _config() -> dict[str, list[object]]:
    return {
        "256-448": [0.6, [[4, 2, 1]]],
        "176-304": [0.4, [[4, 3, 1]]],
    }


def test_sampler_uses_official_resolution_t_batch_encoding() -> None:
    sampler = VariableVideoBatchSampler(_TinyVideoDataset(), _config(), seed=17, shuffle=False)

    batches = list(sampler)

    assert batches
    assert len(sampler) == len(batches)
    assert all(len(batch) in {2, 3} for batch in batches)
    encoded = {
        tuple(int(value) for value in index.split("-"))
        for batch in batches
        for index in batch
    }
    assert encoded
    assert {values[1] for values in encoded} == {4}
    assert {(values[2], values[3]) for values in encoded} <= {(256, 448), (176, 304)}


def test_sampler_is_deterministic_and_rank_lengths_match() -> None:
    rank_zero = VariableVideoBatchSampler(
        _TinyVideoDataset(), _config(), num_replicas=2, rank=0, seed=23
    )
    rank_one = VariableVideoBatchSampler(
        _TinyVideoDataset(), _config(), num_replicas=2, rank=1, seed=23
    )

    first_zero = list(rank_zero)
    first_one = list(rank_one)
    repeat_zero = list(rank_zero)
    repeat_one = list(rank_one)

    assert len(first_zero) == len(first_one) > 0
    assert first_zero == repeat_zero
    assert first_one == repeat_one
    rank_zero.set_epoch(1)
    assert list(rank_zero) != first_zero


def test_sampler_pads_each_rank_to_complete_batches() -> None:
    config = {"256-256": [1.0, [[4, 8, 1]]]}  # one sample, eight per rank

    rank_zero = VariableVideoBatchSampler(
        _TinyVideoDataset(1), config, num_replicas=2, rank=0, shuffle=False
    )
    rank_one = VariableVideoBatchSampler(
        _TinyVideoDataset(1), config, num_replicas=2, rank=1, shuffle=False
    )

    assert list(rank_zero) and list(rank_one)
    assert [len(batch) for batch in rank_zero] == [8]
    assert [len(batch) for batch in rank_one] == [8]
    with pytest.raises(ValueError, match="zero batches"):
        list(
            VariableVideoBatchSampler(
                _TinyVideoDataset(1), config, num_replicas=2, rank=0, drop_last=True
            )
        )


def test_sampler_state_resume_skips_consumed_steps() -> None:
    sampler = VariableVideoBatchSampler(
        _TinyVideoDataset(15),
        {"256-256": [1.0, [[4, 2, 1]]]},
        num_replicas=2,
        rank=0,
        seed=31,
    )
    full_plan = list(sampler)
    state = sampler.state_dict(num_steps=1)
    resumed = VariableVideoBatchSampler(
        _TinyVideoDataset(15),
        {"256-256": [1.0, [[4, 2, 1]]]},
        num_replicas=2,
        rank=0,
        seed=999,
    )
    resumed.load_state_dict(state)

    assert list(resumed) == full_plan[1:]


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"256": [1.0, [[4, 2, 1]]]},
        {"256-256": [1.0, [[0, 2, 1]]]},
        {"256-256": [0.0, [[4, 2, 1]]]},
    ],
)
def test_sampler_rejects_invalid_mix_config(config: dict[object, object]) -> None:
    with pytest.raises(ValueError):
        VariableVideoBatchSampler(_TinyVideoDataset(), config)


def test_loader_factory_builds_batch_sampler_from_mix_config(tmp_path: Path) -> None:
    config_path = tmp_path / "mixed.yaml"
    config_path.write_text(
        """
dataset:
  _class_name: dwm.datasets.common.DatasetAdapter
  base_dataset:
    _class_name: dwm.common.SerializedReadonlyList
    items:
      - images: [[0], [1], [2], [3]]
      - images: [[4], [5], [6], [7]]
  transform_list: []
mix_config:
  256-256: [1.0, [[2, 1, 1]]]
dataloader:
  _class_name: torch.utils.data.DataLoader
  batch_size: 99
  shuffle: true
  num_workers: 0
""",
        encoding="utf-8",
    )

    loader, config = load_dataloader_from_config(config_path)

    assert "mix_config" in config
    assert isinstance(loader.batch_sampler, VariableVideoBatchSampler)
    assert loader.batch_size is None
    batch = next(iter(loader))
    assert len(batch["images"]) == 2
    assert all(len(frame) == 1 for frame in batch["images"])
    assert all(tuple(frame[0].shape) == (1,) for frame in batch["images"])

def test_loader_factory_wraps_invalid_mix_config(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        """
dataset:
  _class_name: dwm.common.SerializedReadonlyList
  items: [0]
mix_config: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="invalid mix_config"):
        load_dataloader_from_config(config_path)
