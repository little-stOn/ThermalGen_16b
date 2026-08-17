"""Generate GLIGEN samples and decode them as native single-channel uint16 TIFF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionGLIGENPipeline
from PIL import Image

from native16_gligen.config import build_common_parser, load_config
from native16_gligen.model_stack import (
    load_checkpoint_layer_into,
    load_model_stack,
    load_stack_into,
    verify_base_model,
)
from native16_gligen.data import load_generation_requests
from native16_gligen.instance_fusion import install_instance_fusion
from native16_gligen.radiometric_bridge import RadiometricLatentBridge
from native16_gligen.radiometric_output_calibration import RadiometricOutputCalibrator
from native16_gligen.radiometric_transport import WindowAwareRadiometricTransport, build_box_mask
from native16_gligen.boxdiff_guidance import (
    CrossAttentionRecorder,
    boxdiff_loss,
    install_recorders,
    phrase_token_groups,
    update_latents,
)
from native16_gligen.native16_vae import (
    Native16LatentAffine,
    assert_clean_official_parent,
    load_native16_vae,
)
from native16_gligen.requests import draw_box_overlay, normalize_request, sanitize_filename
from native16_gligen.style_adapter import adapter_config, load_adapter_checkpoint
from native16_gligen.thermal16 import RadiometricProfile, decode_native16, save_native16_tiff


def preview_image(raw: np.ndarray, profile: RadiometricProfile) -> Image.Image:
    normalized = (raw.astype(np.float32) - profile.window_low) / (
        profile.window_high - profile.window_low
    )
    gray = np.rint(normalized.clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGB")


def quantize_native16_tensor(values: torch.Tensor) -> torch.Tensor:
    """Match the uint16 round-trip used by the validated offline pipeline."""

    return values.float().clamp(0.0, 1.0).mul(65535.0).round().div(65535.0)




def load_radiometric_bridge(checkpoint: Path) -> tuple[RadiometricLatentBridge, dict]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = payload.get("manifest", {})
    if manifest.get("legacy_8bit_checkpoint") is not False:
        raise ValueError(f"radiometric bridge is not marked independent: {checkpoint}")
    state = payload.get("bridge")
    if not isinstance(state, dict):
        raise ValueError(f"radiometric bridge checkpoint has no bridge state: {checkpoint}")
    bridge = RadiometricLatentBridge()
    bridge.load_state_dict(state, strict=True)
    return bridge, manifest


def generate_boxdiff_latents(
    pipe,
    request: dict,
    cfg,
    generator: torch.Generator,
    args,
    recorder: CrossAttentionRecorder,
    original_processors: dict,
    recording_processors: dict,
) -> tuple[torch.Tensor, list[tuple]]:
    """Run the normal GLIGEN loop with latent-only BoxDiff guidance."""

    device = pipe._execution_device
    height, width = int(cfg.model.height), int(cfg.model.width)
    boxes, phrases = normalize_request(request, cfg)
    prompt = str(request.get("prompt", ""))
    negative = str(request.get("negative_prompt", ""))
    guidance = float(
        args.guidance_scale if args.guidance_scale is not None else cfg.sample.guidance_scale
    )
    beta = float(
        args.gligen_beta
        if args.gligen_beta is not None
        else cfg.sample.gligen_scheduled_sampling_beta
    )
    do_cfg = guidance > 1.0
    prompt_embeds, negative_embeds = pipe.encode_prompt(
        prompt, device, 1, do_cfg, negative_prompt=negative
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
    pipe.scheduler.set_timesteps(int(cfg.sample.num_steps), device=device)
    latents = pipe.prepare_latents(
        1,
        pipe.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )
    max_objs = 30
    with torch.no_grad():
        token_inputs = pipe.tokenizer(phrases, padding=True, return_tensors="pt").to(device)
        phrase_embeddings = pipe.text_encoder(**token_inputs).pooler_output
    gligen_boxes = torch.zeros(max_objs, 4, device=device, dtype=pipe.text_encoder.dtype)
    gligen_boxes[: len(boxes)] = torch.tensor(boxes, device=device, dtype=gligen_boxes.dtype)
    positive = torch.zeros(
        max_objs, pipe.unet.config.cross_attention_dim, device=device, dtype=pipe.text_encoder.dtype
    )
    positive[: len(boxes)] = phrase_embeddings
    masks = torch.zeros(max_objs, device=device, dtype=pipe.text_encoder.dtype)
    masks[: len(boxes)] = 1
    gligen_boxes = gligen_boxes[None].expand(1, -1, -1).clone()
    positive = positive[None].expand(1, -1, -1).clone()
    masks = masks[None].expand(1, -1).clone()
    if do_cfg:
        gligen_boxes = torch.cat([gligen_boxes, gligen_boxes])
        positive = torch.cat([positive, positive])
        masks = torch.cat([masks, masks])
        masks[:1] = 0
    cross_kwargs = {
        "gligen": {"boxes": gligen_boxes, "positive_embeddings": positive, "masks": masks}
    }
    num_grounding_steps = int(beta * len(pipe.scheduler.timesteps))
    pipe.enable_fuser(True)
    extra = pipe.prepare_extra_step_kwargs(generator, 0.0)
    groups = phrase_token_groups(pipe.tokenizer, prompt, phrases)
    rows: list[tuple] = []
    for step_index, timestep in enumerate(pipe.scheduler.timesteps):
        if step_index == num_grounding_steps:
            pipe.enable_fuser(False)
        active = (
            int(args.boxdiff_steps) > 0
            and int(args.boxdiff_start) <= step_index
            < int(args.boxdiff_start) + int(args.boxdiff_steps)
        )
        if active:
            recorder.clear()
            recorder.enabled = True
            pipe.unet.set_attn_processor(dict(recording_processors))
            with torch.enable_grad():
                latents = latents.detach().requires_grad_(True)
                model_input = torch.cat([latents, latents]) if do_cfg else latents
                model_input = pipe.scheduler.scale_model_input(model_input, timestep)
                noise = pipe.unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_kwargs,
                ).sample
                attention = recorder.aggregate()
                loss, parts = boxdiff_loss(
                    attention,
                    boxes,
                    groups,
                    float(args.boxdiff_top_p),
                    int(args.boxdiff_corner_radius),
                )
                latents, grad_rms, update_rms = update_latents(
                    latents,
                    loss,
                    float(args.boxdiff_strength),
                    float(args.boxdiff_max_update_rms),
                )
                rows.append(
                    (
                        step_index,
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        grad_rms,
                        update_rms,
                        len(recorder.maps),
                    )
                )
            recorder.enabled = False
            pipe.unet.set_attn_processor(dict(original_processors))
            model_input = torch.cat([latents, latents]) if do_cfg else latents
            model_input = pipe.scheduler.scale_model_input(model_input, timestep)
            with torch.no_grad():
                noise = pipe.unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_kwargs,
                ).sample
        else:
            with torch.no_grad():
                model_input = torch.cat([latents, latents]) if do_cfg else latents
                model_input = pipe.scheduler.scale_model_input(model_input, timestep)
                noise = pipe.unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_kwargs,
                ).sample
        if do_cfg:
            uncond, cond = noise.chunk(2)
            noise = uncond + guidance * (cond - uncond)
        latents = pipe.scheduler.step(noise, timestep, latents, **extra).prev_sample
    pipe.unet.set_attn_processor(dict(original_processors))
    return latents, rows


def main() -> None:
    parser = build_common_parser("Generate native uint16 thermal frames with GLIGEN")
    parser.add_argument(
        "--model-stack",
        type=Path,
        default=None,
        help="Pinned parent model stack; defaults to model.stack_manifest in the recipe",
    )
    parser.add_argument(
        "--checkpoint-layer",
        type=Path,
        default=None,
        help="Newly trained delta whose recorded parent must equal the resolved model stack",
    )
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--gligen-beta", type=float, default=None)
    parser.add_argument("--boxdiff-steps", type=int, default=0)
    parser.add_argument("--boxdiff-strength", type=float, default=0.01)
    parser.add_argument("--boxdiff-start", type=int, default=0)
    parser.add_argument("--boxdiff-top-p", type=float, default=0.2)
    parser.add_argument("--boxdiff-corner-radius", type=int, default=1)
    parser.add_argument("--boxdiff-max-update-rms", type=float, default=0.05)
    parser.add_argument("--boxdiff-attention-res", type=int, default=16)
    parser.add_argument(
        "--latent-calibration",
        type=Path,
        default=None,
        help="Optional reversible Native16 affine calibration in scaled latent space",
    )
    parser.add_argument(
        "--radiometric-output-calibration",
        type=Path,
        default=None,
        help="Optional monotonic output tone calibration JSON",
    )
    parser.add_argument(
        "--radiometric-transport",
        choices=("none", "g065"),
        default=None,
        help="Optional box-aware Native16 output transport preset",
    )
    parser.add_argument(
        "--style-adapter-checkpoint",
        type=Path,
        default=None,
        help="Optional Native16 frequency/mask style adapter applied before transport",
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Save the final scaled diffusion latent for deterministic trajectory training",
    )
    parser.add_argument(
        "--latents-only",
        action="store_true",
        help="Save final latents without decoding or writing TIFF/preview files",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optionally limit the request list for deterministic smoke tests",
    )
    parser.add_argument(
        "--no-previews",
        action="store_true",
        help="Write native TIFF/latents without PNG previews or overlays",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.latents_only:
        args.save_latents = True
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    cfg = load_config(args.cfg, args.set)
    input_mode = str(cfg.data.input_mode).lower()
    if input_mode not in {"native16", "native16_bridge"}:
        raise ValueError("sample_native16.py requires data.input_mode=native16 or native16_bridge")
    stack_path = args.model_stack or getattr(cfg.model, "stack_manifest", None)
    if stack_path in (None, "", "null", "false", False):
        raise ValueError("Native16 generation requires --model-stack or model.stack_manifest")
    parent_stack = load_model_stack(stack_path)
    verify_base_model(parent_stack, str(cfg.model.pretrained_model))
    assert_clean_official_parent(str(cfg.model.pretrained_model))
    requests = load_generation_requests(args.requests)
    if args.max_samples is not None:
        requests = requests[: args.max_samples]
    if args.dry_run:
        print(f"requests={len(requests)} first={requests[0]}")
        return
    device = torch.device("cuda", 0)
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        str(cfg.model.pretrained_model),
        torch_dtype=torch.float16,
        use_safetensors=bool(cfg.model.use_safetensors),
        variant=str(cfg.model.variant) if cfg.model.variant else None,
    ).to(device)
    instance_fusion_layers = install_instance_fusion(
        pipe.unet, getattr(cfg.model, "instance_fusion", None)
    )
    if instance_fusion_layers:
        print(f"installed_instance_fusion_layers={instance_fusion_layers}", flush=True)
    vae_manifest = None
    bridge_manifest = None
    radiometric_bridge = None
    if input_mode == "native16":
        vae, vae_manifest = load_native16_vae(str(cfg.model.vae_pretrained_model))
        pipe.vae = vae.to(device=device, dtype=torch.float32).eval()
    else:
        bridge_path = getattr(cfg.model, "radiometric_bridge_checkpoint", None)
        if bridge_path in (None, "", "null", "false", False):
            raise ValueError("native16_bridge requires model.radiometric_bridge_checkpoint")
        radiometric_bridge, bridge_manifest = load_radiometric_bridge(Path(str(bridge_path)))
        radiometric_bridge = radiometric_bridge.to(device=device, dtype=torch.float32).eval()
        # Keep the official SD VAE, but decode in fp32 for a stable bridge output.
        pipe.vae = pipe.vae.to(device=device, dtype=torch.float32).eval()
        print(f"loaded_radiometric_bridge={bridge_path}", flush=True)
    calibration_path = args.latent_calibration
    if calibration_path is None:
        configured_calibration = getattr(cfg.model, "latent_calibration_path", None)
        if configured_calibration not in (None, "", "null", "false", False):
            calibration_path = Path(str(configured_calibration))
    latent_calibration = None
    if calibration_path is not None:
        latent_calibration = Native16LatentAffine.from_json(calibration_path)
        print(
            "native16_latent_calibration="
            f"{latent_calibration.target} scale={list(latent_calibration.scale)} "
            f"shift={list(latent_calibration.shift)}",
            flush=True,
        )
    output_calibration_path = args.radiometric_output_calibration
    if output_calibration_path is None:
        configured_output_calibration = getattr(cfg.model, "radiometric_output_calibration", None)
        if configured_output_calibration not in (None, "", "null", "false", False):
            output_calibration_path = Path(str(configured_output_calibration))
    output_calibrator = None
    if output_calibration_path is not None:
        output_calibrator = RadiometricOutputCalibrator.from_json(
            output_calibration_path, device=device
        ).eval()
        print(f"loaded_radiometric_output_calibration={output_calibration_path}", flush=True)
    style_adapter_path = args.style_adapter_checkpoint
    if style_adapter_path is None:
        configured_style_adapter = getattr(cfg.model, "style_adapter_checkpoint", None)
        if configured_style_adapter not in (None, "", "null", "false", False):
            style_adapter_path = Path(str(configured_style_adapter))
    style_adapter = None
    style_adapter_payload = None
    if style_adapter_path is not None:
        style_adapter, style_adapter_payload = load_adapter_checkpoint(
            str(style_adapter_path), device
        )
        print(f"loaded_style_adapter={style_adapter_path}", flush=True)
    transport_preset = args.radiometric_transport
    if transport_preset is None:
        configured_transport = getattr(cfg.model, "radiometric_transport", None)
        transport_preset = (
            "none"
            if configured_transport in (None, "", "none", "null", "false", False)
            else str(configured_transport).lower()
        )
    radiometric_transport = None
    if transport_preset != "none":
        if transport_preset == "g065" and style_adapter is None:
            raise ValueError(
                "g065 was validated after the v5 style adapter; configure "
                "model.style_adapter_checkpoint or pass --style-adapter-checkpoint"
            )
        radiometric_transport = WindowAwareRadiometricTransport.from_preset(
            transport_preset
        ).to(device).eval()
        print(f"loaded_radiometric_transport={transport_preset}", flush=True)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    stack_report = load_stack_into(pipe.unet, parent_stack, state_prefix="unet.")
    resolved_model_fingerprint = parent_stack.fingerprint
    print(
        "loaded_model_stack="
        f"{parent_stack.name} fingerprint={parent_stack.fingerprint} "
        f"layers={[(layer.identifier, layer.key_count) for layer in stack_report.layers]}",
        flush=True,
    )
    if args.checkpoint_layer is not None:
        resolved_model_fingerprint = load_checkpoint_layer_into(
            pipe.unet,
            args.checkpoint_layer,
            parent_stack=parent_stack,
            state_prefix="unet.",
        )
        print(
            f"loaded_checkpoint_layer={args.checkpoint_layer} "
            f"fingerprint={resolved_model_fingerprint}",
            flush=True,
        )
    boxdiff_recorder = None
    boxdiff_original_processors = None
    boxdiff_recording_processors = None
    if int(args.boxdiff_steps) > 0:
        boxdiff_recorder = CrossAttentionRecorder(int(args.boxdiff_attention_res))
        boxdiff_original_processors = install_recorders(pipe.unet, boxdiff_recorder)
        boxdiff_recording_processors = dict(pipe.unet.attn_processors)
        pipe.unet.set_attn_processor(dict(boxdiff_original_processors))
        pipe.unet.requires_grad_(False).eval()
        print(
            "boxdiff_guidance="
            f"steps:{args.boxdiff_steps},strength:{args.boxdiff_strength},"
            f"start:{args.boxdiff_start},attention_res:{args.boxdiff_attention_res}",
            flush=True,
        )
    profile = RadiometricProfile.from_json(cfg.data.thermal16.profile_path)
    output = args.output or Path(cfg.sample.output_dir)
    image_dir, preview_dir, overlay_dir = output / "images16", output / "previews", output / "overlays"
    latent_dir = output / "latents"
    directories = [image_dir]
    if not args.no_previews:
        directories.extend((preview_dir, overlay_dir))
    if args.save_latents:
        directories.append(latent_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    base_seed = int(args.seed if args.seed is not None else cfg.sample.seed)
    guidance = float(args.guidance_scale if args.guidance_scale is not None else cfg.sample.guidance_scale)
    beta = float(args.gligen_beta if args.gligen_beta is not None else cfg.sample.gligen_scheduled_sampling_beta)
    records = []
    guidance_log = None
    if int(args.boxdiff_steps) > 0:
        guidance_log = (output / "boxdiff_guidance.tsv").open("w", encoding="utf-8")
        guidance_log.write("name\tstep\tinner\touter\tcorner\tgrad_rms\tupdate_rms\tmaps\n")
    for index, request in enumerate(requests):
        boxes, phrases = normalize_request(request, cfg)
        name = sanitize_filename(str(request.get("name", f"sample_{index:05d}")))
        tiff_path = image_dir / f"{name}.tiff"
        preview_path = preview_dir / f"{name}.png"
        overlay_path = overlay_dir / f"{name}.png"
        latent_path = latent_dir / f"{name}.pt"
        required_paths = (
            (latent_path,)
            if args.latents_only
            else (tiff_path, latent_path)
            if args.save_latents and args.no_previews
            else (tiff_path, preview_path, overlay_path, latent_path)
            if args.save_latents
            else (tiff_path,)
            if args.no_previews
            else (tiff_path, preview_path, overlay_path)
        )
        if args.skip_existing and all(path.is_file() for path in required_paths):
            if args.latents_only:
                records.append({
                    "name": name,
                    "seed": base_seed + index,
                    "latent_path": str(latent_path),
                    "boxes": boxes,
                    "phrases": phrases,
                })
                continue
            with Image.open(tiff_path) as existing:
                existing_raw = np.asarray(existing)
            records.append({
                "name": name,
                "seed": base_seed + index,
                "path": str(tiff_path),
                "dtype": str(existing_raw.dtype),
                "shape": list(existing_raw.shape),
                "minimum": int(existing_raw.min()),
                "maximum": int(existing_raw.max()),
                "latent_path": str(latent_path) if args.save_latents else None,
                "boxes": boxes,
                "phrases": phrases,
            })
            continue
        generator = torch.Generator(device=device).manual_seed(base_seed + index)
        if int(args.boxdiff_steps) > 0:
            latents, boxdiff_rows = generate_boxdiff_latents(
                pipe,
                request,
                cfg,
                generator,
                args,
                boxdiff_recorder,
                boxdiff_original_processors,
                boxdiff_recording_processors,
            )
            if guidance_log is not None:
                for row in boxdiff_rows:
                    guidance_log.write(
                        name
                        + "\t"
                        + "\t".join(
                            f"{value:.7g}" if isinstance(value, float) else str(value)
                            for value in row
                        )
                        + "\n"
                    )
                guidance_log.flush()
        else:
            result = pipe(
                prompt=str(request.get("prompt", "")),
                negative_prompt=str(request.get("negative_prompt", "")),
                gligen_phrases=phrases,
                gligen_boxes=boxes,
                gligen_scheduled_sampling_beta=beta,
                height=int(cfg.model.height),
                width=int(cfg.model.width),
                num_inference_steps=int(cfg.sample.num_steps),
                guidance_scale=guidance,
                generator=generator,
                output_type="latent",
            )
            latents = result.images
        if not torch.is_tensor(latents) or latents.shape[1] != 4:
            raise RuntimeError(f"pipeline did not return BCHW latents: {type(latents)}")
        if args.save_latents:
            torch.save(latents[0].detach().to(device="cpu", dtype=torch.float16), latent_path)
        if args.latents_only:
            records.append({
                "name": name,
                "seed": base_seed + index,
                "latent_path": str(latent_path),
                "latent_shape": list(latents.shape[1:]),
                "boxes": boxes,
                "phrases": phrases,
            })
            continue
        native_latents = (
            latent_calibration.inverse(latents.float())
            if latent_calibration is not None
            else latents.float()
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
            decoded = pipe.vae.decode(
                native_latents / float(pipe.vae.config.scaling_factor), return_dict=True
            ).sample.float()
            if radiometric_bridge is not None:
                values_tensor = radiometric_bridge.output_bridge(decoded)
            else:
                values_tensor = decoded.add(1).div(2).clamp(0, 1)
            if output_calibrator is not None:
                values_tensor = output_calibrator(values_tensor)
            if style_adapter is not None:
                values_tensor = quantize_native16_tensor(values_tensor)
                roi_mask = build_box_mask(
                    boxes,
                    values_tensor.shape[-2],
                    values_tensor.shape[-1],
                    0.25,
                    0.0,
                    "xyxy",
                    values_tensor.device,
                    values_tensor.dtype,
                )
                values_tensor = style_adapter(values_tensor, roi_mask)
                values_tensor = quantize_native16_tensor(values_tensor)
            if radiometric_transport is not None:
                values_tensor = radiometric_transport(
                    values_tensor, boxes, box_format="xyxy"
                )
        values = values_tensor[0, 0].cpu().numpy()
        save_native16_tiff(tiff_path, values, profile)
        raw = decode_native16(values, profile)
        if not args.no_previews:
            preview = preview_image(raw, profile)
            preview.save(preview_path)
            draw_box_overlay(preview, boxes, phrases).save(overlay_path)
        records.append({
            "name": name,
            "seed": base_seed + index,
            "path": str(tiff_path),
            "dtype": "uint16",
            "shape": list(raw.shape),
            "minimum": int(raw.min()),
            "maximum": int(raw.max()),
            "latent_path": str(latent_path) if args.save_latents else None,
            "boxes": boxes,
            "phrases": phrases,
        })
    metadata = {
        "schema_version": 2,
        "branch": input_mode,
        "model_stack": parent_stack.metadata(),
        "checkpoint_layer": (
            str(args.checkpoint_layer.resolve()) if args.checkpoint_layer is not None else None
        ),
        "resolved_model_fingerprint": resolved_model_fingerprint,
        "vae_manifest": vae_manifest,
        "bridge_manifest": bridge_manifest,
        "latent_calibration": (
            {
                "target": latent_calibration.target,
                "scale": list(latent_calibration.scale),
                "shift": list(latent_calibration.shift),
                "source_vae": latent_calibration.source_vae,
                "path": str(calibration_path.resolve()),
            }
            if latent_calibration is not None
            else None
        ),
        "radiometric_output_calibration": (
            str(output_calibration_path.resolve())
            if output_calibration_path is not None
            else None
        ),
        "style_adapter": (
            {
                "checkpoint": str(style_adapter_path.resolve()),
                "step": style_adapter_payload.get("step"),
                "parameters": adapter_config(style_adapter),
                "mask_expand": 0.25,
            }
            if style_adapter is not None and style_adapter_payload is not None
            else None
        ),
        "radiometric_transport": (
            {
                "preset": transport_preset,
                "box_format": "xyxy",
                "parameters": radiometric_transport.config.to_dict(),
            }
            if radiometric_transport is not None
            else None
        ),
        "guidance_scale": guidance,
        "gligen_beta": beta,
        "previews_written": not args.no_previews,
        "latents_only": bool(args.latents_only),
        "records": records,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if guidance_log is not None:
        guidance_log.close()
        pipe.unet.set_attn_processor(dict(boxdiff_original_processors))
    print(f"saved={len(records)} native_uint16_tiff={image_dir}")


if __name__ == "__main__":
    main()
