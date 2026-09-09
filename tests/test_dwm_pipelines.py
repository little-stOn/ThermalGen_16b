from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from dwm.common import load_pipeline_from_config  # noqa: E402
from dwm.pipelines import (  # noqa: E402
    ConditionBundle,
    PipelineContractError,
    ThermalVideoBatch,
    ThermalVideoPipeline,
)


class TinyVideoModel(torch.nn.Module):
    """Small test double for the model contract, not a production model."""

    latent_dtype = torch.float32

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def encode_video(self, batch: ThermalVideoBatch) -> torch.Tensor:
        if batch.images is None:
            raise AssertionError("test model requires images during training")
        return batch.images.float()

    def decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        return latents.sigmoid()

    def latent_shape(self, batch: ThermalVideoBatch) -> tuple[int, ...]:
        return (*batch.leading_shape, 1, 2, 2)

    def sample_timesteps(self, batch: ThermalVideoBatch, generator: Any) -> torch.Tensor:
        return torch.zeros(batch.batch_size, dtype=torch.long, device=batch.images.device)

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return latents + noise

    def training_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return noise

    def prepare_conditions(
        self,
        batch: ThermalVideoBatch,
        *,
        training: bool,
        generator: Any,
    ) -> ConditionBundle:
        return ConditionBundle("conditional", None if training else "unconditional")

    def forward_video(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        conditions: Any,
        batch: ThermalVideoBatch,
    ) -> torch.Tensor:
        offset = 0.1 if conditions == "unconditional" else 0.0
        return noisy_latents * self.weight + offset

    def inference_timesteps(self, num_steps: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        return tuple(torch.tensor(index, device=device) for index in range(num_steps, 0, -1))

    def scheduler_step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        generator: Any,
    ) -> torch.Tensor:
        return latents - 0.1 * model_output



class DeterministicVideoModel(TinyVideoModel):
    """Deterministic test double for accumulation weighting."""

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return latents

    def training_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(latents)

def make_batch(batch_size: int = 2, time_steps: int = 2) -> dict[str, Any]:
    sample_ids = [
        [[f"butiv:scene-{index}:{time}:view" ] for time in range(time_steps)]
        for index in range(batch_size)
    ]
    boxes = [
        [[torch.zeros((0, 4), dtype=torch.float32)] for _ in range(time_steps)]
        for _ in range(batch_size)
    ]
    labels = [[[()] for _ in range(time_steps)] for _ in range(batch_size)]
    return {
        "vae_images": torch.full(
            (batch_size, time_steps, 1, 1, 2, 2), 0.5, dtype=torch.float32
        ),
        "box_condition_images": torch.zeros(
            (batch_size, time_steps, 1, 3, 2, 2), dtype=torch.float32
        ),
        "boxes": boxes,
        "labels": labels,
        "track_ids": labels,
        "sample_ids": sample_ids,
        "bbox_available": torch.ones((batch_size, time_steps, 1), dtype=torch.bool),
        "condition_valid": torch.zeros((batch_size, time_steps, 1), dtype=torch.bool),
        "pts": torch.arange(time_steps, dtype=torch.float32).view(1, time_steps, 1).repeat(batch_size, 1, 1),
        "pts_unit": "seconds",
        "fps": torch.full((batch_size,), 10.0),
        "box_image_sizes": torch.full((batch_size, time_steps, 1, 2), 2.0),
    }

def test_loader_batch_is_mapped_without_losing_nested_annotations() -> None:
    prepared = ThermalVideoBatch.from_loader(make_batch())

    assert prepared.leading_shape == (2, 2, 1)
    assert tuple(prepared.images.shape) == (2, 2, 1, 1, 2, 2)
    assert tuple(prepared.box_condition_images.shape) == (2, 2, 1, 3, 2, 2)
    assert prepared.box_coordinate_space == "pixel"
    assert tuple(prepared.box_image_sizes.shape) == (2, 2, 1, 2)
    assert prepared.pts_unit == "seconds"
    assert prepared.sample_ids[0][0][0].startswith("butiv:")

def test_loader_timestamps_are_normalized_to_seconds() -> None:
    batch = make_batch()
    batch["pts"] = batch["pts"] * 1000.0
    batch["pts_unit"] = ["milliseconds", "milliseconds"]

    prepared = ThermalVideoBatch.from_loader(batch)

    assert prepared.pts_unit == "seconds"
    assert torch.equal(prepared.pts[0], torch.tensor([[0.0], [1.0]]))


def test_batch_rejects_mismatched_condition_spatial_shape() -> None:
    batch = make_batch()
    batch["box_condition_images"] = torch.zeros((2, 2, 1, 3, 4, 2))

    with pytest.raises(PipelineContractError, match=r"share \[H,W\]"):
        ThermalVideoBatch.from_loader(batch)


def test_batch_defaults_undeclared_timestamp_units_to_seconds() -> None:
    batch = make_batch()
    batch.pop("pts_unit")

    prepared = ThermalVideoBatch.from_loader(batch)

    assert prepared.pts_unit == "seconds"
    assert torch.equal(prepared.pts, batch["pts"])


def test_train_step_accumulates_and_flushes_at_real_boundary() -> None:
    model = TinyVideoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pipeline = ThermalVideoPipeline(model, optimizer=optimizer, accumulation_steps=2, seed=7)

    first = pipeline.train_step(make_batch())
    assert first["updated"] is False
    assert first["optimizer_step"] == 0

    second = pipeline.train_step(make_batch())
    assert second["updated"] is True
    assert second["optimizer_step"] == 1
    assert pipeline.flush() is False


def test_accumulation_weights_variable_batch_counts() -> None:
    model = DeterministicVideoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pipeline = ThermalVideoPipeline(model, optimizer=optimizer, accumulation_steps=2)

    first = make_batch(batch_size=1)
    first["vae_images"].fill_(0.25)
    second = make_batch(batch_size=3)
    second["vae_images"].fill_(0.75)

    pipeline.train_step(first)
    pipeline.train_step(second)

    assert model.weight.item() == pytest.approx(0.228125)


def test_inference_jointly_returns_decoded_video_and_supports_cfg() -> None:
    model = TinyVideoModel()
    pipeline = ThermalVideoPipeline(model, seed=11)

    result = pipeline.inference_pipeline(
        make_batch(), num_steps=3, guidance_scale=2.0
    )

    assert tuple(result.latents.shape) == (2, 2, 1, 1, 2, 2)
    assert tuple(result.frames.shape) == (2, 2, 1, 1, 2, 2)
    assert bool(torch.isfinite(result.frames).all())
    assert result.num_steps == 3


def test_inference_can_use_condition_only_batch() -> None:
    batch = make_batch()
    batch.pop("vae_images")
    model = TinyVideoModel()
    pipeline = ThermalVideoPipeline(model, seed=13)

    result = pipeline.generate(batch, num_steps=1)

    assert tuple(result.latents.shape[:3]) == (2, 2, 1)
    assert result.frames is not None


def test_checkpoint_restores_model_optimizer_and_counters(tmp_path: Path) -> None:
    model = TinyVideoModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pipeline = ThermalVideoPipeline(model, optimizer=optimizer, seed=17)
    pipeline.train_step(make_batch())
    pipeline.flush()
    checkpoint = tmp_path / "video.pt"
    pipeline.save_checkpoint(checkpoint, metadata={"topology": "test"})

    restored_model = TinyVideoModel()
    restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
    restored = ThermalVideoPipeline(restored_model, optimizer=restored_optimizer, seed=99)
    metadata = restored.load_checkpoint(checkpoint)

    assert metadata == {"topology": "test"}
    assert restored.micro_step == pipeline.micro_step
    assert restored.optimizer_step == pipeline.optimizer_step
    assert torch.equal(restored_model.weight, model.weight)


def test_pipeline_can_be_constructed_from_config_with_runtime_model(tmp_path: Path) -> None:
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        "pipeline:\n"
        "  _class_name: dwm.pipelines.ThermalVideoPipeline\n"
        "  accumulation_steps: 2\n",
        encoding="utf-8",
    )

    pipeline, config = load_pipeline_from_config(
        config_path,
        model=TinyVideoModel(),
    )

    assert isinstance(pipeline, ThermalVideoPipeline)
    assert pipeline.accumulation_steps == 2
    assert config["pipeline"]["_class_name"] == "dwm.pipelines.ThermalVideoPipeline"
