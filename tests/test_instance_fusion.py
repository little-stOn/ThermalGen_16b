from types import SimpleNamespace

import torch
from torch import nn

from native16_gligen.instance_fusion import (
    InstanceFusionContext,
    ScaleAwareInstanceFusion,
    build_soft_instance_masks,
    compute_scale_weights,
    install_instance_fusion,
)


def test_soft_mask_is_zero_outside_box() -> None:
    boxes = torch.tensor([[[0.25, 0.25, 0.75, 0.75]]])
    valid = torch.tensor([[True]])
    mask = build_soft_instance_masks(boxes, valid, side=16)[0, 0]
    assert torch.count_nonzero(mask[:4]).item() == 0
    assert torch.count_nonzero(mask[:, :4]).item() == 0
    assert mask[8, 8] > 0


def test_scale_routing_separates_small_and_large() -> None:
    boxes = torch.tensor(
        [[[0.10, 0.10, 0.14, 0.14], [0.10, 0.10, 0.60, 0.60]]]
    )
    valid = torch.tensor([[True, True]])
    low = compute_scale_weights(boxes, valid, feature_side=8)
    high = compute_scale_weights(boxes, valid, feature_side=64)
    assert high[0, 0] > low[0, 0]
    assert low[0, 1] > high[0, 1]


def test_zero_init_and_spatial_support() -> None:
    adapter = ScaleAwareInstanceFusion(query_dim=16, object_dim=12, rank=8)
    hidden = torch.randn(1, 64, 16)
    objects = torch.randn(1, 2, 12)
    context = InstanceFusionContext(
        boxes=torch.tensor([[[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.0, 0.0]]]),
        masks=torch.tensor([[True, False]]),
    )
    assert torch.equal(adapter(hidden, objects, context), torch.zeros_like(hidden))
    nn.init.normal_(adapter.output.weight, std=0.02)
    output = adapter(hidden, objects, context).view(1, 8, 8, 16)
    assert torch.count_nonzero(output[:, :2]).item() == 0
    assert torch.count_nonzero(output[:, :, :2]).item() == 0
    assert torch.count_nonzero(output[:, 3:5, 3:5]).item() > 0


class PositionNet(nn.Module):
    def forward(self, boxes, positive_embeddings, masks):
        return positive_embeddings * masks[..., None]


class GatedSelfAttentionDense(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(12, 16)
        self.alpha_attn = nn.Parameter(torch.tensor(0.0))
        self.alpha_dense = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, objs):
        return x


class MockUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.position_net = PositionNet()
        self.fuser = GatedSelfAttentionDense()


def test_install_preserves_old_parameter_names_and_output() -> None:
    unet = MockUNet()
    old_keys = set(unet.state_dict())
    installed = install_instance_fusion(
        unet,
        SimpleNamespace(
            enabled=True,
            rank=8,
            feather_power=2.0,
            preferred_extent=3.0,
            log_sigma=1.25,
            minimum_scale_weight=0.05,
        ),
    )
    assert installed == 1
    assert old_keys.issubset(set(unet.state_dict()))
    boxes = torch.tensor([[[0.2, 0.2, 0.8, 0.8]]])
    valid = torch.tensor([[1.0]])
    objects = unet.position_net(
        boxes=boxes, positive_embeddings=torch.randn(1, 1, 12), masks=valid
    )
    hidden = torch.randn(1, 64, 16)
    assert torch.equal(unet.fuser(hidden, objects), hidden)
