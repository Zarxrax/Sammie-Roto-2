"""Multi-GPU DFR (Diffusion Fidelity Rendering) runner.
Runs :class:`DFRPipeline` across multiple GPUs with:
- **Stage 1 and stage 2 (detailing)** -- sequence parallelism (SP) on both
  ``stage`` and ``stage_detailing`` (including temporal tiles and the spatial
  epilogue, which reuse those two objects)
- **Gemma** -- Accelerate-based parallelization
- **VAE** -- distributed decoding
Requires ``ltx-kernels`` to be installed (transitive via SP builder).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from functools import wraps
from multiprocessing import SimpleQueue
from typing import Any

import torch
import torch.distributed as dist

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import ModelRegistry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import AUTO_TILING, AutoTiling, TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.multigpu.transformer.attention import AttentionManager
from ltx_core.multigpu.vae.distributed_decoder import dist_rank_tile_pixel_shape
from ltx_core.quantization import QuantizationPolicy
from ltx_core.quantization.fp8_cast import build_policy as _build_fp8_cast_policy
from ltx_core.tiling import DimensionTilingConfig, TileCountConfig, balanced_tile_split
from ltx_core.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape
from ltx_pipelines.dfr_pipeline import DFRPipeline, add_dfr_cli_args
from ltx_pipelines.multigpu.controller import MGPUController
from ltx_pipelines.multigpu.gemma_builders import AccelerateGemmaBuilder
from ltx_pipelines.multigpu.runner import MGPURunner
from ltx_pipelines.multigpu.sp_builder import SequenceParallelBuilder
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_pipelines.multigpu.weight_tracker import TransformerWeightTracker
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration

logger = logging.getLogger(__name__)

# Stage 2 at 4K dominates. 3840x2176 (4K UHD height snapped to the 32x VAE grid)
# with 301 pixel frames pads to a 313-frame DFR canvas (F_lat=40, 13 keyframe
# slots) plus the x2 IC-LoRA half-res reference:
#   40*68*120 + 13*68*120 + 40*34*60 = 514080 video tokens.
_DEFAULT_SP_MAX_TOKENS = 524288
# Rank that collects distributed-VAE tiles and encodes the assembled video.
_DRIVER_RANK = 0
# Temporal batches the driver assembles the decoded canvas in. One batch per rank (the Dist
# default) is 129 frames of 3840x2176 RGB bf16 -- 6.5 GiB on top of a driver already holding
# every rank's decoded tile. 16 keeps a batch near 1.7 GiB; the encoder consumes them one at a time.
_DEFAULT_VAE_TEMPORAL_BATCHES = 16


@contextmanager
def _dfr_dist_tile_auto_tiling(
    vae_tiling: TileCountConfig,
    *,
    rank: int,
    world_size: int,
    scale: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
) -> Iterator[None]:
    """Rewrite ``DFRPipeline`` AUTO recommend to this rank's Dist-tile pixel shape.
    Patches ``ltx_pipelines.dfr_pipeline.ensure_tiling_config`` (the name
    ``DFRPipeline.__call__`` looks up) for the duration of the ``with`` block.
    Dist stays a pass-through; the pipeline still returns the resolved
    ``TilingConfig``. Explicit configs and ``None`` are unchanged.
    """
    import ltx_pipelines.dfr_pipeline as dfr_mod  # noqa: PLC0415

    orig = dfr_mod.ensure_tiling_config

    @wraps(orig)
    def ensure_tiling_config_dist_tile(
        tiling_config: TilingConfig | AutoTiling | None,
        *,
        video_shape: VideoPixelShape,
        scale_factors: SpatioTemporalScaleFactors,
        vae_checkpoint_path: str,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
        device: torch.device | None = None,
        free_bytes: int | None = None,
        keyframes: bool = False,
    ) -> TilingConfig | None:
        if isinstance(tiling_config, AutoTiling):
            latent_shape = VideoLatentShape.from_pixel_shape(video_shape, scale_factors=scale)
            tile_shape = dist_rank_tile_pixel_shape(
                vae_tiling,
                latent_shape,
                scale,
                rank,
                world_size,
                fps=video_shape.fps,
                batch=video_shape.batch,
            )
            logger.info(
                "rank=%s AUTO_TILING Dist tile video_shape=%s (full canvas %s)",
                rank,
                tile_shape,
                video_shape,
            )
            video_shape = tile_shape
        return orig(
            tiling_config,
            video_shape=video_shape,
            scale_factors=scale_factors,
            vae_checkpoint_path=vae_checkpoint_path,
            diffvae_optimization=diffvae_optimization,
            device=device,
            free_bytes=free_bytes,
            keyframes=keyframes,
        )

    dfr_mod.ensure_tiling_config = ensure_tiling_config_dist_tile
    try:
        yield
    finally:
        dfr_mod.ensure_tiling_config = orig


class DFRRunner(MGPURunner):
    """Distributed :class:`DFRPipeline`: SP on both stages + Accelerate Gemma + distributed VAE."""

    _vae_tiling: TileCountConfig | None = None

    @torch.inference_mode()
    def setup(  # noqa: PLR0913
        self,
        *,
        model_paths: ModelPaths,
        spatial_upsampler_path: str,
        vae_queue: SimpleQueue,
        detailing_lora: Sequence[LoraPathStrengthAndSDOps],
        prompt_enhancer_gemma_root: str | None = None,
        temporal_upsampler_path: str | None = None,
        loras: Sequence[LoraPathStrengthAndSDOps] = (),
        compilation_config: CompilationConfig | None = None,
        sp_max_tokens: int = _DEFAULT_SP_MAX_TOKENS,
        quantization: Callable[[], QuantizationPolicy] | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
        vae_temporal_batches: int = _DEFAULT_VAE_TEMPORAL_BATCHES,
    ) -> None:
        # quantization is a picklable zero-arg builder (built per worker, post-spawn); default fp8-cast.
        quantization_policy = (
            quantization() if quantization is not None else _build_fp8_cast_policy(model_paths.transformer())
        )
        registry = ModelRegistry(cache_models=True, cache_weights=True)
        pipeline = DFRPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=spatial_upsampler_path,
            loras=list(loras),
            detailing_lora=list(detailing_lora),
            temporal_upsampler_path=temporal_upsampler_path,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
            registry=registry,
            quantization=quantization_policy,
            compilation_config=compilation_config,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
            diffvae_optimization=diffvae_optimization,
        )
        tracker = TransformerWeightTracker(group=self.groups.transformer_group)

        # SP on both stages: stage 1 / temporal rounds share ``stage``; stage 2 /
        # the spatial epilogue use ``stage_detailing`` (same checkpoint, extra IC-LoRA).
        model_cfg = pipeline.stage._transformer_builder.model_config().get("transformer", {})
        attn_mgr = AttentionManager(
            max_tokens=sp_max_tokens,
            num_heads=model_cfg["num_attention_heads"],
            head_dim=model_cfg["attention_head_dim"],
            tensor_dtype=pipeline.dtype,
            group=self.groups.transformer_group,
        )
        for stage in (pipeline.stage, pipeline.stage_detailing):
            stage._transformer_builder = SequenceParallelBuilder(
                inner=stage._transformer_builder,
                attn_mgr=attn_mgr,
                registry=registry,
                tracker=tracker,
            )

        # Accelerate Gemma parallelization. Capture shared-vs-separate before replacing
        # the encode builder so a shared alias is re-bound to the new instance.
        pe = pipeline.prompt_encoder
        separate_enhancer = pe._enhancer_text_encoder_builder is not pe._text_encoder_builder
        pe._text_encoder_builder = AccelerateGemmaBuilder(
            gemma_root_path=model_paths.text_encoder(),
            gemma_group=self.groups.gemma_group,
            broadcast_group=self.groups.transformer_group,
            registry=registry,
            src_rank=_DRIVER_RANK,
            dtype=pipeline.dtype,
        )
        if separate_enhancer:
            assert prompt_enhancer_gemma_root is not None
            pe._enhancer_text_encoder_builder = AccelerateGemmaBuilder(
                gemma_root_path=prompt_enhancer_gemma_root,
                gemma_group=self.groups.gemma_group,
                broadcast_group=self.groups.transformer_group,
                registry=registry,
                src_rank=_DRIVER_RANK,
                dtype=pipeline.dtype,
            )
        else:
            pe._enhancer_text_encoder_builder = pe._text_encoder_builder

        # Distributed VAE decoding: balanced 2D spatial grid over the group (one tile/rank).
        # height takes the smaller factor of world_size, width the larger; size-aware split is a follow-up.
        vae_height_tiles, vae_width_tiles = balanced_tile_split(dist.get_world_size(self.groups.vae_group))
        vae_tiling = TileCountConfig(
            height=DimensionTilingConfig(num_tiles=vae_height_tiles, overlap=4),
            width=DimensionTilingConfig(num_tiles=vae_width_tiles, overlap=4),
        )
        pipeline.video_decoder._decoder_builder = DistributedDecoderBuilder(  # type: ignore[assignment]
            inner=pipeline.video_decoder._decoder_builder,
            queue=vae_queue,
            vae_group=self.groups.vae_group,
            vae_tiling=vae_tiling,
            driver_rank=_DRIVER_RANK,
            registry=registry,
            num_temporal_batches=vae_temporal_batches,
        )
        self._vae_tiling = vae_tiling
        self._pipeline = pipeline

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        *,
        output_path: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        images: list[Any] | None = None,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        temporal_upscalings: int = 0,
        spatial_upscalings: int = 1,
    ) -> Iterator[str | None]:
        # The pipeline raises ValueError on invalid input (symmetric across ranks); the controller
        # catches that and turns it into a recoverable RunnerError. Anything else is fatal.
        vae_tiling = self._vae_tiling
        auto_tile = (
            _dfr_dist_tile_auto_tiling(
                vae_tiling,
                rank=dist.get_rank(self.groups.vae_group),
                world_size=dist.get_world_size(self.groups.vae_group),
            )
            if vae_tiling is not None
            else nullcontext()
        )
        with auto_tile:
            result = self._pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images or [],
                tiling_config=AUTO_TILING,
                enhance_prompt=enhance_prompt,
                enhance_static_cache=enhance_static_cache,
                temporal_upscalings=temporal_upscalings,
                spatial_upscalings=spatial_upscalings,
                decode_device_fn=lambda _: self._pipeline.device,
            )
        if dist.get_rank() != _DRIVER_RANK:
            yield None  # workers: nothing to encode
            return
        encode_video(
            video=result.video,
            fps=int(frame_rate * (2**temporal_upscalings)),
            audio=result.audio,
            output_path=output_path,
            video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        )
        yield output_path


if __name__ == "__main__":
    from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser, resolve_cli_params

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    params = resolve_cli_params(distilled=True)
    parser = add_dfr_cli_args(default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True))
    args = parser.parse_args()

    vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    controller = MGPUController(DFRRunner)
    controller.start(
        model_paths=args.model_paths,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        vae_queue=vae_queue,
        detailing_lora=args.detailing_lora,
        temporal_upsampler_path=args.temporal_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        compilation_config=args.compile,
        diffvae_optimization=args.diffvae_optimization,
    )
    try:
        for _ in controller.stream(
            output_path=args.output_path,
            prompt=args.prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            images=args.images,
            enhance_prompt=args.enhance_prompt,
            enhance_static_cache=args.enhance_static_cache,
            temporal_upscalings=args.temporal_upscalings,
            spatial_upscalings=args.spatial_upscalings,
        ):
            pass  # drive the job to completion; the runner writes the file as a side effect
    finally:
        controller.shutdown()
