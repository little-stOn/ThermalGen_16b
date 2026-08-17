from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer
from native16_gligen.instance_fusion import install_instance_fusion
from native16_gligen.radiometric_bridge import RadiometricLatentBridge
from native16_gligen.radiometric_output_calibration import RadiometricOutputCalibrator

from native16_gligen.native16_vae import (
    Native16LatentAffine,
    assert_clean_official_parent,
    load_native16_vae,
)


@dataclass
class EncodedText:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


class GLIGENDenoisingCore(nn.Module):
    """Diffusers GLIGEN UNet wrapper used by both training and inference."""

    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_hidden_states: torch.Tensor,
        boxes: torch.Tensor,
        phrase_embeddings: torch.Tensor,
        box_mask: torch.Tensor,
    ) -> torch.Tensor:
        gligen = {
            "boxes": boxes,
            "positive_embeddings": phrase_embeddings,
            "masks": box_mask.to(dtype=boxes.dtype),
        }
        return self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_hidden_states,
            cross_attention_kwargs={"gligen": gligen},
            return_dict=True,
        ).sample


@dataclass
class DiffusionSystem:
    vae: AutoencoderKL
    tokenizer: CLIPTokenizer
    text_encoder: CLIPTextModel
    noise_scheduler: DDPMScheduler
    inference_scheduler: DDIMScheduler
    core: GLIGENDenoisingCore
    device: torch.device
    radiometric_bridge: RadiometricLatentBridge | None = None
    radiometric_output_calibrator: RadiometricOutputCalibrator | None = None
    latent_calibration: Native16LatentAffine | None = None

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        dtype = next(self.vae.parameters()).dtype
        with torch.no_grad():
            if self.radiometric_bridge is not None:
                if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
                    raise ValueError("native16_bridge expects three fixed radiometric channels")
                features = pixel_values.add(1.0).div(2.0).to(self.device, dtype=torch.float32)
                vae_input = self.radiometric_bridge.input_bridge(features)
                posterior = self.vae.encode(vae_input.to(dtype=dtype)).latent_dist
                latents = posterior.mode() * float(self.vae.config.scaling_factor)
            else:
                posterior = self.vae.encode(pixel_values.to(self.device, dtype=dtype)).latent_dist
                latents = posterior.sample() * float(self.vae.config.scaling_factor)
        if self.latent_calibration is not None:
            latents = self.latent_calibration.apply(latents)
        return latents

    def decode_latents_with_grad(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode x0 latents while keeping gradients to the denoiser output.

        VAE weights stay frozen, but the decoder Jacobian must remain in the
        graph so pixel-space detail losses can update the grounding modules.
        Decoding is forced to fp32 even when the denoiser uses fp16 autocast.
        """

        dtype = next(self.vae.parameters()).dtype
        if self.latent_calibration is not None:
            latents = self.latent_calibration.inverse(latents)
        scaled = latents.float() / float(self.vae.config.scaling_factor)
        autocast_device = self.device.type if self.device.type in {"cuda", "cpu"} else "cuda"
        with torch.autocast(device_type=autocast_device, enabled=False):
            decoded = self.vae.decode(scaled.to(dtype), return_dict=True).sample
        if self.radiometric_bridge is not None:
            decoded = self.radiometric_bridge.output_bridge(decoded)
            if self.radiometric_output_calibrator is not None:
                decoded = self.radiometric_output_calibrator(decoded)
            return decoded.float().mul(2.0).sub(1.0).clamp(-1.0, 1.0)
        return decoded.float()

    def encode_text(self, prompts: list[str]) -> EncodedText:
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device)
        attention_mask = tokens.attention_mask.to(self.device)
        with torch.no_grad():
            hidden = self.text_encoder(input_ids, attention_mask=attention_mask)[0]
        return EncodedText(hidden.float(), attention_mask.bool())

    def encode_phrases(self, phrases: list[list[str]], box_mask: torch.Tensor) -> torch.Tensor:
        batch_size, max_boxes = box_mask.shape
        flat = [phrase for row in phrases for phrase in row]
        if len(flat) != batch_size * max_boxes:
            raise ValueError("Each sample must provide exactly data.max_boxes padded phrases")
        tokens = self.tokenizer(flat, padding=True, truncation=True, return_tensors="pt")
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            pooled = self.text_encoder(**tokens, return_dict=True).pooler_output
        embeddings = pooled.reshape(batch_size, max_boxes, -1).float()
        return embeddings * box_mask.to(embeddings.dtype).unsqueeze(-1)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = str(name).lower()
    if normalized == "fp32":
        return torch.float32
    if normalized == "fp16":
        return torch.float16
    if normalized == "bf16":
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 8:
            raise ValueError("bf16 is unavailable on V100/Volta. Use fp16 or fp32")
        return torch.bfloat16
    raise ValueError(f"Unknown dtype name: {name}")


def _pretrained_kwargs(cfg: Any, include_variant: bool = True) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"use_safetensors": bool(getattr(cfg.model, "use_safetensors", True))}
    if getattr(cfg.model, "revision", None):
        kwargs["revision"] = cfg.model.revision
    if include_variant and getattr(cfg.model, "variant", None):
        kwargs["variant"] = cfg.model.variant
    return kwargs


def _freeze(module: nn.Module, device: torch.device, dtype: torch.dtype) -> None:
    module.requires_grad_(False)
    module.eval()
    module.to(device=device, dtype=dtype)


def configure_trainable(core: GLIGENDenoisingCore, mode: str) -> dict[str, int]:
    core.requires_grad_(False)
    matched = 0
    for name, parameter in core.unet.named_parameters():
        grounding = "position_net" in name or ".fuser." in name
        grounding_adapter = (
            "position_net" in name
            or name.endswith(".fuser.alpha_attn")
            or name.endswith(".fuser.alpha_dense")
        )
        crossattn = "attn2.to_k" in name or "attn2.to_v" in name
        if mode == "full_unet":
            selected = True
        elif mode == "instance_fusion":
            selected = ".instance_fusion." in name
        elif mode == "style_unet":
            selected = not grounding
        elif mode == "grounding_crossattn":
            selected = grounding or crossattn
        elif mode == "grounding_adapter":
            selected = grounding_adapter
        else:
            selected = grounding
        parameter.requires_grad = selected
        if selected:
            matched += parameter.numel()
    if mode not in {
        "grounding_only",
        "grounding_adapter",
        "grounding_crossattn",
        "instance_fusion",
        "full_unet",
        "style_unet",
    }:
        raise ValueError(f"Unsupported training mode: {mode}")
    if matched == 0:
        raise RuntimeError("No trainable GLIGEN parameters found; check the supplied UNet and mode")
    total = sum(parameter.numel() for parameter in core.parameters())
    return {"trainable": matched, "frozen": total - matched, "total": total}


def build_system(cfg: Any, device: torch.device) -> DiffusionSystem:
    model_id = str(cfg.model.pretrained_model)
    input_mode = str(getattr(cfg.data, "input_mode", "rgb8")).lower()
    text_dtype = resolve_dtype(str(cfg.model.text_encoder_precision), device)
    vae_dtype = resolve_dtype(str(cfg.model.vae_precision), device)
    # Keep trainable parameters in fp32; train.precision controls autocast. PyTorch
    # GradScaler cannot safely unscale gradients whose parameter storage is fp16.
    model_dtype = torch.float32
    common = _pretrained_kwargs(cfg)
    metadata = _pretrained_kwargs(cfg, include_variant=False)
    # The 16-bit branch may point at an independently fine-tuned VAE exported
    # with save_pretrained().  Leaving this unset preserves the original
    # 8-bit SD/GLIGEN behaviour byte-for-byte.
    vae_model_id = str(getattr(cfg.model, "vae_pretrained_model", "") or model_id)
    vae_subfolder = getattr(cfg.model, "vae_subfolder", "vae")
    # VAE exports produced by train_thermal16_vae.py use the standard
    # diffusion_pytorch_model.safetensors name, independently of the UNet
    # fp16 variant used by the GLIGEN checkpoint.
    latent_calibration: Native16LatentAffine | None = None
    radiometric_bridge: RadiometricLatentBridge | None = None
    radiometric_output_calibrator: RadiometricOutputCalibrator | None = None
    if input_mode == "native16":
        assert_clean_official_parent(model_id)
        if vae_subfolder not in (None, "", "null", "vae"):
            raise ValueError("native16 VAE must use its validated vae subdirectory")
        vae, _ = load_native16_vae(vae_model_id)
        calibration_path = getattr(cfg.model, "latent_calibration_path", None)
        if calibration_path not in (None, "", "null", "false", False):
            latent_calibration = Native16LatentAffine.from_json(str(calibration_path))
    elif input_mode == "native16_bridge":
        assert_clean_official_parent(model_id)
        bridge_path = getattr(cfg.model, "radiometric_bridge_checkpoint", None)
        if bridge_path in (None, "", "null", "false", False):
            raise ValueError("native16_bridge requires model.radiometric_bridge_checkpoint")
        payload = torch.load(str(bridge_path), map_location="cpu", weights_only=False)
        manifest = payload.get("manifest", {})
        if manifest.get("legacy_8bit_checkpoint") is not False:
            raise ValueError("native16 bridge checkpoint is not marked independent")
        radiometric_bridge = RadiometricLatentBridge()
        radiometric_bridge.load_state_dict(payload["bridge"], strict=True)
        vae_kwargs = {"torch_dtype": vae_dtype}
        if getattr(cfg.model, "revision", None):
            vae_kwargs["revision"] = cfg.model.revision
        vae_kwargs["subfolder"] = "vae"
        vae = AutoencoderKL.from_pretrained(model_id, **vae_kwargs)
    else:
        vae_kwargs = {"torch_dtype": vae_dtype}
        if getattr(cfg.model, "revision", None):
            vae_kwargs["revision"] = cfg.model.revision
        if getattr(cfg.model, "vae_use_safetensors", None) is not None:
            vae_kwargs["use_safetensors"] = bool(cfg.model.vae_use_safetensors)
        if vae_subfolder not in (None, "", "null"):
            vae_kwargs["subfolder"] = str(vae_subfolder)
        vae = AutoencoderKL.from_pretrained(vae_model_id, **vae_kwargs)
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **metadata)
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=text_dtype, **common
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=model_dtype, **common
    )
    if getattr(unet, "position_net", None) is None:
        raise ValueError("Configured checkpoint has no GLIGEN position_net; use a converted GLIGEN SD1.4 checkpoint")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    inference_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    _freeze(vae, device, vae_dtype)
    if radiometric_bridge is not None:
        _freeze(radiometric_bridge, device, torch.float32)
    calibration_path = getattr(cfg.model, "radiometric_output_calibration", None)
    if calibration_path not in (None, "", "null", "false", False):
        radiometric_output_calibrator = RadiometricOutputCalibrator.from_json(
            str(calibration_path), device=device
        )
        _freeze(radiometric_output_calibrator, device, torch.float32)
        print(f"loaded_radiometric_output_calibration={calibration_path}", flush=True)
    _freeze(text_encoder, device, text_dtype)
    unet.to(device=device, dtype=model_dtype)
    if latent_calibration is not None:
        print(
            "native16_latent_calibration="
            f"{latent_calibration.target} scale={list(latent_calibration.scale)} "
            f"shift={list(latent_calibration.shift)}",
            flush=True,
        )
    core = GLIGENDenoisingCore(unet)
    instance_fusion_layers = install_instance_fusion(
        unet, getattr(cfg.model, "instance_fusion", None)
    )
    if instance_fusion_layers:
        print(f"installed_instance_fusion_layers={instance_fusion_layers}", flush=True)
    return DiffusionSystem(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        noise_scheduler=noise_scheduler,
        inference_scheduler=inference_scheduler,
        core=core,
        device=device,
        latent_calibration=latent_calibration,
        radiometric_bridge=radiometric_bridge,
        radiometric_output_calibrator=radiometric_output_calibrator,
    )
