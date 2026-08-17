"""Training-free BoxDiff-style spatial guidance for Diffusers GLIGEN.

The model is always frozen.  Gradients are taken only with respect to the
current latent during an early denoising step, then immediately detached.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GuidanceStats:
    inner: float
    outer: float
    corner: float
    grad_rms: float
    update_rms: float
    maps: int


class CrossAttentionRecorder:
    """Records selected conditional text cross-attention probabilities."""

    def __init__(self, attention_res: int = 16) -> None:
        self.attention_res = attention_res
        self.maps: list[torch.Tensor] = []
        self.enabled = False

    def clear(self) -> None:
        self.maps.clear()

    def add(self, probabilities: torch.Tensor, batch_size: int) -> None:
        # probabilities: [batch * heads, query, text_tokens]
        query = probabilities.shape[1]
        if not self.enabled or query != self.attention_res * self.attention_res:
            return
        heads = probabilities.shape[0] // batch_size
        maps = probabilities.reshape(batch_size, heads, query, probabilities.shape[-1])
        # CFG batches are [unconditional, conditional].  Keep the conditional map.
        self.maps.append(maps[-1].mean(dim=0).reshape(self.attention_res, self.attention_res, -1))

    def aggregate(self) -> torch.Tensor:
        if not self.maps:
            raise RuntimeError(
                f"No {self.attention_res}x{self.attention_res} text cross-attention maps were recorded"
            )
        return torch.stack(self.maps).mean(dim=0)


class RecordingAttnProcessor:
    """Diffusers AttnProcessor equivalent that additionally saves attention maps."""

    def __init__(self, recorder: CrossAttentionRecorder) -> None:
        self.recorder = recorder

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        probabilities = attn.get_attention_scores(query, key, attention_mask)
        if is_cross:
            self.recorder.add(probabilities, batch_size)
        hidden_states = torch.bmm(probabilities, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_recorders(unet, recorder: CrossAttentionRecorder) -> dict[str, object]:
    """Replace only attn2 processors; restore with ``unet.set_attn_processor``."""
    originals = dict(unet.attn_processors)
    processors = {
        name: RecordingAttnProcessor(recorder) if name.endswith("attn2.processor") else processor
        for name, processor in originals.items()
    }
    if processors == originals:
        raise RuntimeError("No cross-attention processors matched '*.attn2.processor'")
    # Diffusers consumes processor dictionaries with ``pop`` while installing.
    unet.set_attn_processor(dict(processors))
    return originals


def phrase_token_groups(tokenizer, prompt: str, phrases: Iterable[str]) -> dict[tuple[int, ...], list[int]]:
    """Map boxes to prompt token spans, grouping repeated phrases safely."""
    prompt_ids = tokenizer(prompt, add_special_tokens=True).input_ids
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for box_index, phrase in enumerate(phrases):
        candidates = [str(phrase), str(phrase).replace(" ", "_")]
        span: tuple[int, ...] | None = None
        for candidate in candidates:
            phrase_ids = tokenizer(candidate, add_special_tokens=True).input_ids[1:-1]
            for start in range(1, len(prompt_ids) - len(phrase_ids)):
                if prompt_ids[start : start + len(phrase_ids)] == phrase_ids:
                    span = tuple(range(start, start + len(phrase_ids)))
                    break
            if span is not None:
                break
        if span is None:
            raise ValueError(f"Could not map GLIGEN phrase {phrase!r} to prompt tokens: {prompt!r}")
        groups[span].append(box_index)
    return dict(groups)


def _box_mask(boxes: list[list[float]], indices: list[int], height: int, width: int, device) -> torch.Tensor:
    mask = torch.zeros((height, width), device=device, dtype=torch.float32)
    for index in indices:
        x1, y1, x2, y2 = boxes[index]
        left, right = sorted((int(round(x1 * width)), int(round(x2 * width))))
        top, bottom = sorted((int(round(y1 * height)), int(round(y2 * height))))
        left, right = max(0, left), min(width, max(left + 1, right))
        top, bottom = max(0, top), min(height, max(top + 1, bottom))
        mask[top:bottom, left:right] = 1
    return mask


def boxdiff_loss(
    attention: torch.Tensor,
    boxes: list[list[float]],
    token_groups: dict[tuple[int, ...], list[int]],
    top_p: float,
    corner_radius: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Original BoxDiff-style inner, outer and corner terms at one attention scale.

    Repeated prompt tokens use a union for the outer constraint. This prevents a
    car in one requested box being penalised as the "outside" of another car.
    """
    height, width, _ = attention.shape
    # The UNet is loaded in fp16 for the fixed sampling protocol.  Reapplying
    # BoxDiff's ``softmax(100 * A)`` to already-softmaxed fp16 probabilities
    # saturates and produces an exactly-zero latent gradient.  Keep the
    # differentiable probabilities, cast before arithmetic, and normalize only
    # by the per-map maximum for comparable top-k terms.
    token_attention = attention.float()
    token_attention = token_attention / token_attention.amax(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    inner_terms, outer_terms, corner_terms = [], [], []
    for span, indices in token_groups.items():
        image = token_attention[:, :, list(span)].mean(dim=-1)
        union = _box_mask(boxes, indices, height, width, image.device)
        outside = 1.0 - union
        k_out = max(1, int(round(float(outside.sum()) * top_p)))
        outer_terms.append((image * outside).reshape(-1).topk(k_out).values.mean())
        for index in indices:
            mask = _box_mask(boxes, [index], height, width, image.device)
            k_in = max(1, int(round(float(mask.sum()) * top_p)))
            inner_terms.append(1.0 - (image * mask).reshape(-1).topk(k_in).values.mean())
            x1, y1, x2, y2 = boxes[index]
            left, right = sorted((int(round(x1 * width)), int(round(x2 * width))))
            top, bottom = sorted((int(round(y1 * height)), int(round(y2 * height))))
            target_x = torch.zeros(width, device=image.device)
            target_y = torch.zeros(height, device=image.device)
            target_x[max(0, left):min(width, max(left + 1, right))] = 1
            target_y[max(0, top):min(height, max(top + 1, bottom))] = 1
            focus_x = torch.zeros(width, device=image.device)
            focus_y = torch.zeros(height, device=image.device)
            for coordinate, focus, limit in ((left, focus_x, width), (right, focus_x, width), (top, focus_y, height), (bottom, focus_y, height)):
                focus[max(0, coordinate - corner_radius):min(limit, coordinate + corner_radius + 1)] = 1
            corner_terms.append(
                (F.l1_loss(image.max(dim=0).values, target_x, reduction="none") * focus_x).mean()
                + (F.l1_loss(image.max(dim=1).values, target_y, reduction="none") * focus_y).mean()
            )
    inner = torch.stack(inner_terms).mean()
    outer = torch.stack(outer_terms).mean()
    corner = torch.stack(corner_terms).mean()
    return inner + outer + corner, (inner, outer, corner)


def update_latents(
    latents: torch.Tensor,
    loss: torch.Tensor,
    strength: float,
    max_update_rms: float,
) -> tuple[torch.Tensor, float, float]:
    gradient = torch.autograd.grad(loss, latents, retain_graph=False, create_graph=False)[0]
    grad_rms = gradient.float().square().mean().sqrt()
    if not torch.isfinite(grad_rms) or grad_rms.item() == 0:
        raise RuntimeError(f"Invalid BoxDiff latent gradient RMS: {grad_rms.item()}")
    delta = -float(strength) * gradient / grad_rms.to(gradient.dtype)
    delta_rms = delta.float().square().mean().sqrt()
    if delta_rms > max_update_rms:
        delta = delta * (float(max_update_rms) / delta_rms).to(delta.dtype)
        delta_rms = delta.float().square().mean().sqrt()
    return (latents + delta).detach(), float(grad_rms), float(delta_rms)
