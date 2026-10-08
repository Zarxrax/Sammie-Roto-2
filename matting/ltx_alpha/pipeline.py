"""Lean stage-1 LTX 2.5 IC-LoRA pipeline with cached text conditioning."""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import torch
from safetensors import safe_open


class LTXAlphaPipeline:
    def __init__(self, transformer, video_vae, lora, empty_context, device):
        # colour-science may leave an optional SciPy placeholder in sys.modules.
        # Transformers probes module specs and rejects that placeholder; removing
        # it correctly reports SciPy as unavailable without adding the dependency.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message='"SciPy" related API features are not available.*')
            warnings.filterwarnings("ignore", message='"Matplotlib" related API features are not available.*')
            import colour  # noqa: F401
        for module_name in ("scipy", "matplotlib"):
            module = sys.modules.get(module_name)
            if module is not None and getattr(module, "__spec__", None) is None:
                del sys.modules[module_name]
        try:
            from ltx_core.conditioning import VideoConditionByReferenceLatent
            from ltx_core.model.video_vae import AUTO_TILING
            from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, VideoDecoder
            from ltx_pipelines.utils.model_paths import ModelPaths
            from matting.ltx_alpha.gguf_runtime import GGUFTransformerBuilder
        except ImportError as exc:
            raise RuntimeError("LTX runtime is not installed. Run the Sammie installer again.") from exc

        self.device = torch.device(device)
        # MPS cannot reliably mix BF16 activations with the transformer's FP32
        # normalization parameters. Keep transformer and VAE activations in
        # FP32 on Metal while the large base weights remain packed Q4. The MPS
        # DiffVAE encoder can silently emit NaNs in FP16.
        self.dtype = torch.float32 if self.device.type == "mps" else torch.bfloat16
        self.encoder_dtype = torch.float32 if self.device.type == "mps" else torch.bfloat16
        self.decoder_dtype = torch.float16 if self.device.type == "mps" else torch.bfloat16
        self._reference_type = VideoConditionByReferenceLatent
        self._auto_tiling = AUTO_TILING
        self._paths = ModelPaths.from_split(
            transformer_path=str(transformer), video_vae_path=str(video_vae))
        self.video_context = self._load_context(empty_context, transformer)
        self.image_conditioner = (
            None if self.device.type == "mps"
            else ImageConditioner(self._paths.video_vae(), self.encoder_dtype, self.device)
        )
        builder = GGUFTransformerBuilder(
            str(Path(transformer).resolve()), str(Path(lora).resolve()))
        self.stage = DiffusionStage(builder, self.dtype, self.device)
        self.video_decoder = VideoDecoder(
            self._paths.video_vae(), self.decoder_dtype, self.device)

    def _load_context(self, path, transformer):
        with safe_open(str(path), framework="pt", device="cpu") as source:
            metadata = source.metadata() or {}
            for key in ("video_context", "video_prompt_embeds"):
                if key in source.keys():
                    value = source.get_tensor(key)
                    return value.unsqueeze(0) if value.ndim == 2 else value
            if "conditioning.0" in source.keys():
                if metadata.get("format") != "comfyui-lossless-conditioning":
                    raise ValueError(
                        f"{path} contains conditioning.0 but is not a ComfyUI lossless conditioning cache")
                try:
                    structure = json.loads(metadata["structure"])
                    options = dict(structure[0]["options"]["items"])
                except (KeyError, TypeError, ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid lossless conditioning metadata in {path}") from exc
                if options.get("unprocessed_ltxav_embeds") is not True:
                    raise ValueError(
                        "The cached LTX conditioning must retain unprocessed_ltxav_embeds=True")
                from matting.ltx_alpha.gguf_runtime import process_cached_conditioning
                # The small prompt connector triggers an MPS autocast bug when
                # its attention mixes BF16 and FP32 positional values. Run this
                # one-time startup conversion on CPU; the result is kept on CPU
                # until it is moved to the transformer device for denoising.
                connector_device = (
                    torch.device("cpu") if self.device.type == "mps" else self.device
                )
                return process_cached_conditioning(
                    str(transformer), source.get_tensor("conditioning.0"),
                    connector_device,
                    torch.bfloat16 if self.device.type == "mps" else self.dtype,
                )
        raise KeyError(f"No video_context tensor in {path}")

    @torch.inference_mode()
    def __call__(self, frames, fps, seed):
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.conditioning import VideoConditionByReferenceLatent
        from ltx_core.types import VideoPixelShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import (
            cleanup_memory,
            create_initial_video_latent,
            ensure_tiling_config,
            tiling_scale_factors_for_vae,
        )
        from ltx_pipelines.utils.types import ModalitySpec, VideoAudio

        class FiniteCheckingDenoiser(SimpleDenoiser):
            def __call__(self, transformer, video_state, audio_state, sigmas, step_index):
                result = super().__call__(
                    transformer, video_state, audio_state, sigmas, step_index)
                if result.video is not None and not torch.isfinite(result.video.denoised).all():
                    raise RuntimeError(
                        f"LTX Alpha produced non-finite values during denoising step "
                        f"{step_index + 1} of {len(sigmas) - 1}."
                    )
                return result

        array = np.asarray(frames, dtype=np.float32) / 127.5 - 1.0
        original_h, original_w = array.shape[1:3]
        pad_h = (-original_h) % 32
        pad_w = (-original_w) % 32
        if pad_h or pad_w:
            array = np.pad(array, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        height, width = array.shape[1:3]
        reference = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0)
        reference = reference.to(device=self.device, dtype=self.encoder_dtype)
        num_frames = len(frames)
        if (num_frames - 1) % 8:
            raise ValueError(f"LTX frame count must be 8n+1, got {num_frames}")

        tiling = ensure_tiling_config(
            self._auto_tiling,
            scale_factors=tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path),
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(1, num_frames, height, width, fps),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        if self.device.type == "mps":
            from matting.ltx_alpha.diffusers_encoder import encode_reference
            latent = encode_reference(
                self._paths.video_vae(), reference, self.device, self.encoder_dtype)
            conditionings = [VideoConditionByReferenceLatent(
                latent.to(dtype=self.dtype), downscale_factor=1,
                temporal_scale_factor=1, strength=1.0)]
        else:
            def encode(video_encoder):
                latent = video_encoder.tiled_encode(reference, tiling) if tiling else video_encoder(reference)
                return [VideoConditionByReferenceLatent(
                    latent.to(device=self.device, dtype=self.dtype), downscale_factor=1,
                    temporal_scale_factor=1, strength=1.0)]
            conditionings = self.image_conditioner(encode)
        if not torch.isfinite(conditionings[0].latent).all():
            raise RuntimeError("LTX Alpha VAE encoding produced non-finite reference latents.")
        video_latent = create_initial_video_latent(
            width=width,
            height=height,
            frames=num_frames,
            fps=fps,
            device=self.device,
            dtype=self.dtype,
            scale_factors=self.stage.video_scale_factors,
        )
        context = self.video_context.to(device=self.device, dtype=self.dtype)
        sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        state, _ = self.stage(
            denoiser=FiniteCheckingDenoiser(context, None),
            sigmas=sigmas,
            noiser=GaussianNoiser(generator=generator),
            modalities=VideoAudio(video=ModalitySpec(
                latent=video_latent,
                conditioning_fps=fps,
                context=context,
                conditionings=conditionings,
                noise_scale=float(sigmas[0]),
            )),
        )
        if not torch.isfinite(state.latent).all():
            raise RuntimeError(
                "LTX Alpha denoising produced non-finite values. "
                "No matte was written; try a different seed or a smaller processing resolution."
            )
        decoded = torch.cat(list(self.video_decoder(
            state.latent, tiling, generator=generator, dtype=self.decoder_dtype)), dim=0)
        if not torch.isfinite(decoded).all():
            raise RuntimeError(
                "LTX Alpha VAE decoding produced non-finite values. "
                "No matte was written; try a smaller processing resolution."
            )
        result = decoded[:, :original_h, :original_w, :3].float().cpu().numpy()
        cleanup_memory()
        return np.clip(result.mean(axis=-1), 0.0, 1.0)
