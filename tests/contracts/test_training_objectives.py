from __future__ import annotations

import pytest
import torch

from native16_gligen.objectives import (
    bounded_contribution,
    compose_total_loss,
    pixel_target_for_decoded,
    style_target,
)


def test_auxiliary_objectives_are_additive() -> None:
    diffusion = torch.tensor(4.0, requires_grad=True)
    first = bounded_contribution(
        "instance",
        torch.tensor(2.0, requires_grad=True),
        diffusion,
        nominal_weight=0.5,
        warmup_fraction=1.0,
        max_ratio=1.0,
    )
    second = bounded_contribution(
        "detail",
        torch.tensor(3.0, requires_grad=True),
        diffusion,
        nominal_weight=0.25,
        warmup_fraction=1.0,
        max_ratio=1.0,
    )
    total = compose_total_loss(diffusion, [first, second])

    assert total.item() == pytest.approx(5.75)
    total.backward()
    assert diffusion.grad is not None


def test_native16_style_keeps_full_triplet_while_detail_uses_absolute() -> None:
    decoded = torch.zeros(2, 1, 8, 8)
    target = torch.randn(2, 3, 8, 8)

    detail = pixel_target_for_decoded(decoded, target, "native16_bridge")
    style = style_target(target, "native16_bridge")

    assert detail.shape == (2, 1, 8, 8)
    assert torch.equal(detail, target[:, :1])
    assert torch.equal(style, target)


def test_style_target_rejects_missing_radiometric_channels() -> None:
    with pytest.raises(ValueError, match=r"\[B,3,H,W\]"):
        style_target(torch.zeros(1, 1, 4, 4), "native16_bridge")
