"""Distributed video decoder that partitions the latent across ranks.
Tiles are assigned to ranks via round-robin, so the number of tiles
may exceed the number of GPUs (e.g. 16 tiles on 4 GPUs = 4 tiles per
rank).  Each rank decodes its assigned tiles sequentially.  Workers
put their list of decoded tiles into a ``mp.Queue`` (CUDA IPC —
zero-copy handle sharing).  The driver collects all tiles, blends
overlap zones, and returns temporal batches distributed across devices.
``decode_single_frames`` splits a *list of independent one-frame clips* instead of one
volume: the planes go round-robin to the ranks and come back through an all-gather, so
each is decoded once and every rank still holds the whole list.
The tiling configuration comes from ``MGPUConfig.vae_tiling`` (set at
construction time), NOT from the pipeline's SGPU tiling kwarg.  MGPU
tiling controls parallelism; SGPU tiling controls single-GPU VRAM
management — they are independent concerns.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from einops import rearrange
from torch.multiprocessing import Queue

from ltx_core.model.disposable import Disposable
from ltx_core.model.video_vae.keyframes import DecodeKeyframes
from ltx_core.model.video_vae.video_vae import (
    VideoDecoder,
    clip_generators,
    iter_decoded_single_frames,
    map_spatial_slice,
    map_temporal_slice,
    to_mapping_operation,
    validate_single_frame_latents,
)
from ltx_core.tiling import (
    Tile,
    TileCountConfig,
    TilingConfig,
    compute_summed_weights,
    create_tiles,
    masks_are_complementary,
    scale_by_masks_1d,
)
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape

logger = logging.getLogger(__name__)


def _sgpu_has_temporal_tiling(tiling_config: TilingConfig | None, num_frames: int) -> bool:
    """True when SGPU tiling would produce more than one temporal chunk for ``num_frames``."""
    return tiling_config is not None and tiling_config.video_chunks_number(num_frames) > 1


def create_distributed_tiles(
    vae_tiling: TileCountConfig,
    latent_shape: VideoLatentShape,
    scale: SpatioTemporalScaleFactors,
) -> list[Tile]:
    """Latent tiles Dist decode uses: ``vae_tiling`` splitters + latent→pixel mappers."""
    t_split, h_split, w_split = vae_tiling.to_splitters(scale, causal_temporal=True)
    return create_tiles(
        torch.Size([latent_shape.frames, latent_shape.height, latent_shape.width]),
        splitters=[t_split, h_split, w_split],
        mappers=[
            to_mapping_operation(map_temporal_slice, scale.time),
            to_mapping_operation(map_spatial_slice, scale.height),
            to_mapping_operation(map_spatial_slice, scale.width),
        ],
    )


def dist_rank_tiles(
    vae_tiling: TileCountConfig,
    latent_shape: VideoLatentShape,
    scale: SpatioTemporalScaleFactors,
    rank: int,
    world_size: int,
) -> list[Tile]:
    """Round-robin Dist tiles for ``rank`` (same assignment as :meth:`DistributedVideoDecoder._decode_tiles`)."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    all_tiles = create_distributed_tiles(vae_tiling, latent_shape, scale)
    return [t for i, t in enumerate(all_tiles) if i % world_size == rank]


def dist_rank_plane_indices(count: int, rank: int, world_size: int) -> list[int]:
    """Round-robin share of ``count`` independent single-frame clips owned by ``rank``.
    The same assignment rule as :func:`dist_rank_tiles`, on a list of whole clips instead of
    tiles of one volume. Fewer clips than ranks, or a ragged tail, simply leaves the trailing
    ranks with a shorter (possibly empty) share -- there is nothing to pad, because every rank
    still joins the gather that follows.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return list(range(rank, count, world_size))


def dist_rank_tile_pixel_shape(
    vae_tiling: TileCountConfig,
    latent_shape: VideoLatentShape,
    scale: SpatioTemporalScaleFactors,
    rank: int,
    world_size: int,
    *,
    fps: float,
    batch: int = 1,
) -> VideoPixelShape:
    """Pixel extent of this rank's Dist decode volume (largest tile if the rank owns several).
    Matches ``decode_video`` of each ``latent[:, :, in_coords]`` slice: spatial axes scale
    by ``scale.height`` / ``scale.width``; temporal is ``(F_lat - 1) * scale.time + 1``.
    """
    my_tiles = dist_rank_tiles(vae_tiling, latent_shape, scale, rank, world_size)
    if not my_tiles:
        raise ValueError(f"rank {rank}/{world_size} has no Dist tiles")

    def pixel_thw(tile: Tile) -> tuple[int, int, int]:
        f_lat = tile.in_coords[0].stop - tile.in_coords[0].start
        h_lat = tile.in_coords[1].stop - tile.in_coords[1].start
        w_lat = tile.in_coords[2].stop - tile.in_coords[2].start
        return (f_lat - 1) * scale.time + 1, h_lat * scale.height, w_lat * scale.width

    frames, height, width = max(map(pixel_thw, my_tiles), key=lambda thw: thw[0] * thw[1] * thw[2])
    return VideoPixelShape(batch=batch, frames=frames, height=height, width=width, fps=fps)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedTile:
    """A VAE-decoded tile with pixel-space placement.
    Attributes:
        pixels: ``[F_tile, H_tile, W_tile, C]`` in the decoder's native dtype.
        pixel_tile: Carries ``out_coords`` (f, h, w slices) and separable ``masks_1d``.
    """

    pixels: torch.Tensor
    pixel_tile: Tile


# ------------------------------------------------------------------
# Tile construction helpers
# ------------------------------------------------------------------


def _to_decoded_tile(
    pixels: torch.Tensor,
    tile: Tile,
) -> DecodedTile:
    """Wrap ``decode_video`` output ``[F, H, W, C]`` in ``[0, 1]`` as a :class:`DecodedTile`."""
    return DecodedTile(pixels=pixels, pixel_tile=tile)


def gather_frames(
    tiles: list[DecodedTile],
    total_frames: int,
    output_height: int,
    output_width: int,
    num_temporal_batches: int,
    world_size: int,
    weights: torch.Tensor | None = None,
    device_fn: Callable[[int], str | torch.device] | None = None,
) -> Iterator[torch.Tensor]:
    """Assemble decoded tiles into temporal batches distributed across GPUs.
    Each temporal batch is allocated on the device returned by *device_fn(batch_index)*.
    By default batches are placed round-robin on ``cuda:0`` … ``cuda:<world_size-1>``.
    Blending multiplies by separable ``mf*mh*mw``. When *weights* is ``None``, masks
    are assumed complementary and the denominator pass is skipped; otherwise
    ``output`` is divided by *weights*.
    """
    if device_fn is None:
        device_fn = lambda b: f"cuda:{b % world_size}"  # noqa: E731

    batch_size = (total_frames + num_temporal_batches - 1) // num_temporal_batches

    for b in range(num_temporal_batches):
        batch_range = slice(b * batch_size, min((b + 1) * batch_size, total_frames))
        batch_len = batch_range.stop - batch_range.start
        if batch_len <= 0:
            break

        device = device_fn(b)
        dtype = tiles[0].pixels.dtype
        output = torch.zeros(batch_len, output_height, output_width, 3, device=device, dtype=dtype)

        for tile in tiles:
            f_slice, h_slice, w_slice = tile.pixel_tile.out_coords

            overlap = slice(max(batch_range.start, f_slice.start), min(batch_range.stop, f_slice.stop))
            if overlap.start >= overlap.stop:
                continue

            tile_frames = slice(overlap.start - f_slice.start, overlap.stop - f_slice.start)
            out_frames = slice(overlap.start - batch_range.start, overlap.stop - batch_range.start)

            # Blend weights stay float32 so the multiply promotes bf16/fp16 pixels.
            mf, mh, mw = (m.to(device=device, dtype=torch.float32) for m in tile.pixel_tile.masks_1d)
            mf = mf[tile_frames]
            pix = tile.pixels[tile_frames].to(device=device, non_blocking=True)
            # Channel axis is not on the pixel Tile; length-1 ones broadcasts over C.
            output[out_frames, h_slice, w_slice, :] += scale_by_masks_1d(
                pix, (mf, mh, mw, torch.ones(1, device=device, dtype=torch.float32))
            )

        if weights is not None:
            batch_weights = weights[batch_range.start : batch_range.stop].to(device=device)
            output.div_(batch_weights[:, :, :, None])
        yield output


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class DistributedVideoDecoder(torch.nn.Module, Disposable):
    """Distributed VAE decoder with queue-based tile collection.
    Wraps any :class:`~ltx_core.model.video_vae.video_vae.VideoDecoder`
    (conv or diffusion) and decodes each MGPU tile via public
    ``decode_video`` (already ``[F, H, W, C]`` in ``[0, 1]``). Full-video
    assembly is :func:`gather_frames` on the driver.
    All ranks decode their latent tile in parallel.  Workers send
    their :class:`DecodedTile` to the driver rank via the shared
    ``mp.Queue`` (CUDA IPC — zero-copy).  The driver collects all
    tiles, blends overlapping regions, and returns temporal batches
    as an iterator.
    Parameters
    ----------
    decoder:
        The real (local) ``VideoDecoder`` instance.
    queue:
        ``mp.Queue`` shared across all ranks for CUDA IPC tile transfer.
    vae_group:
        NCCL process group for the VAE ranks. Used to derive
        ``rank`` and ``world_size`` within the group.
    vae_tiling:
        MGPU tiling config that determines how the latent is split.
    driver_rank:
        Group-local rank of the driver process (the rank that collects
        and assembles tiles).
    num_temporal_batches:
        Number of temporal batches the driver assembles the canvas in
        (:func:`gather_frames`); each is a ``frames/n x H x W x 3`` buffer, so a
        higher count lowers peak driver VRAM. Defaults to one batch per rank.
    """

    def __init__(
        self,
        decoder: VideoDecoder,
        queue: Queue,  # type: ignore[type-arg]
        vae_group: dist.ProcessGroup,
        vae_tiling: TileCountConfig,
        driver_rank: int = 0,
        num_temporal_batches: int | None = None,
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.queue = queue
        self.vae_group = vae_group
        self.rank = dist.get_rank(vae_group)
        self.world_size = dist.get_world_size(vae_group)
        self.vae_tiling = vae_tiling
        self.driver_rank = driver_rank
        if num_temporal_batches is not None and num_temporal_batches < 1:
            raise ValueError(f"num_temporal_batches must be >= 1, got {num_temporal_batches}")
        self.num_temporal_batches = self.world_size if num_temporal_batches is None else num_temporal_batches

    @property
    def video_downscale_factors(self) -> SpatioTemporalScaleFactors:
        return self.decoder.video_downscale_factors

    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Non-tiled path: full local ``decode_video``, returned as ``[B, C, F, H, W]``.
        Uses the public decoder API (Conv: single-pass; Diff: full Euler loop)
        rather than ``decoder.forward``, which is one diffusion step on Diff.
        """
        chunks = list(self.decoder.decode_video(sample, tiling_config=None, generator=generator))
        pixels = torch.cat(chunks, dim=0)  # [F, H, W, C] in [0, 1]
        return rearrange(pixels, "f h w c -> 1 c f h w")

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        keyframes: DecodeKeyframes | None = None,
        device_fn: Callable[[int], str | torch.device] | None = None,
        **_kwargs: object,
    ) -> Iterator[torch.Tensor]:
        """Distributed decode — all ranks decode, driver assembles.
        Not a generator so that worker side-effects (decode + queue.put)
        execute eagerly regardless of whether the caller iterates.
        1. Each rank decodes its latent tile (with optional intra-GPU tiling).
        2. Workers send their :class:`DecodedTile` to the driver via the queue.
        3. The driver collects all tiles, blends overlaps, and returns
           temporal batches distributed across GPUs.
        """
        latent_shape = VideoLatentShape.from_torch_shape(latent.shape)
        scale = self.decoder.video_downscale_factors
        full_shape = latent_shape.upscale(scale)

        # Phase 1: each rank decodes its assigned tiles.
        my_tiles = self._decode_tiles(latent, latent_shape, scale, generator, tiling_config, keyframes)

        # Phase 2: workers send tiles to driver.
        if self.rank != self.driver_rank:
            self.queue.put((self.rank, my_tiles))
            return iter([])

        # Phase 3: driver collects and assembles.
        all_tiles = self._collect_tiles(my_tiles)
        if masks_are_complementary(
            [t.pixel_tile for t in all_tiles],
            (full_shape.frames, full_shape.height, full_shape.width),
        ):
            weights = None
        else:
            logger.warning(
                "VAE blend masks are not complementary; falling back to dense [F,H,W] "
                "weight normalization (expensive for large videos)."
            )
            weights = compute_summed_weights(
                [t.pixel_tile for t in all_tiles],
                (full_shape.frames, full_shape.height, full_shape.width),
            )
        batches = gather_frames(
            all_tiles,
            full_shape.frames,
            full_shape.height,
            full_shape.width,
            self.num_temporal_batches,
            self.world_size,
            weights,
            device_fn=device_fn,
        )
        return batches

    def decode_single_frames(
        self,
        latents: Sequence[torch.Tensor],
        generator: torch.Generator | Sequence[torch.Generator | None] | None = None,
    ) -> Iterator[torch.Tensor]:
        """Split the one-frame clips round-robin over the group, then all-gather the pixels.
        Each plane is an independent clip, so there is nothing to tile and nothing to blend:
        rank ``r`` decodes planes ``r, r + W, r + 2W, ...`` on the inner SGPU decoder and every
        rank ends up holding the whole list, in the order the latents were given. A DiffVAE
        plane is expensive enough that decoding all of them on every rank -- what this used to
        do -- wastes ``W - 1`` GPUs per plane.
        ``decode_video`` is still the wrong tool here: it splits one volume, and its workers
        yield nothing after queueing their tile.
        Not a generator. The all-gather has to run on every rank whether or not the caller
        iterates the result, and a rank whose share is empty (fewer planes than ranks, or a
        ragged tail) still takes part with an empty payload instead of skipping the collective
        -- one absent participant blocks the group until the NCCL timeout.
        Pixels cross the group as CPU tensors: ``all_gather_object`` pickles a CUDA tensor with
        its source device index, so a plane decoded on ``cuda:3`` would be rebuilt on ``cuda:3``
        on every rank. Each plane is moved onto the device of its own latent on the way out, and
        a rank keeps the GPU copy of the planes it decoded itself.
        With a per-plane ``generator`` sequence the pixels do not depend on the world size. A
        single shared generator (or ``None``) is drawn from independently on each rank, so with
        those the noise a given plane gets does.
        """
        if not latents:
            return iter(())
        validate_single_frame_latents(latents)
        gens = clip_generators(len(latents), generator)
        if self.world_size == 1:
            return iter_decoded_single_frames(self.decoder, latents, gens)
        if len(latents) > 1 and not isinstance(generator, Sequence):
            logger.warning(
                "Dist single-frame decode is splitting %d planes over %d ranks with a shared "
                "generator; each rank draws its own noise, so the pixels depend on the world "
                "size. Pass one generator per latent to keep them world-size invariant.",
                len(latents),
                self.world_size,
            )

        my_indices = dist_rank_plane_indices(len(latents), self.rank, self.world_size)
        logger.info(
            "Dist single-frame decode: rank %d of %d takes %d of %d planes",
            self.rank,
            self.world_size,
            len(my_indices),
            len(latents),
        )
        decoded = iter_decoded_single_frames(
            self.decoder, [latents[i] for i in my_indices], [gens[i] for i in my_indices]
        )
        mine: dict[int, torch.Tensor] = dict(zip(my_indices, decoded, strict=True))

        payload = [(index, pixels.to("cpu")) for index, pixels in mine.items()]
        shares: list[list[tuple[int, torch.Tensor]] | None] = [None] * self.world_size
        dist.all_gather_object(shares, payload, group=self.vae_group)
        merged: dict[int, torch.Tensor] = {}
        for share in shares:
            merged.update(dict(share or ()))
        # This rank's own planes are already on the right device; prefer them over the round trip.
        merged.update(mine)
        missing = [index for index in range(len(latents)) if index not in merged]
        if missing:
            raise RuntimeError(f"Dist single-frame decode lost planes {missing} in the all-gather")
        # Lazy on the way out: the collective already ran, and the caller usually consumes one
        # plane at a time, so only the ranks' own planes stay resident on the GPU.
        return (merged[index].to(latents[index].device) for index in range(len(latents)))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode_tiles(
        self,
        latent: torch.Tensor,
        latent_shape: VideoLatentShape,
        scale: SpatioTemporalScaleFactors,
        generator: torch.Generator | None,
        tiling_config: TilingConfig | None = None,
        keyframes: DecodeKeyframes | None = None,
    ) -> list[DecodedTile]:
        """Decode this rank's assigned latent tiles and convert to :class:`DecodedTile` list.
        Each tile gets its keyframe planes narrowed to its own window:
        * spatially, with the same slices as its video latent -- planes are full-frame, so
          handing them over uncropped offsets every one of them by the tile origin;
        * temporally, via :meth:`DecodeKeyframes.for_frame_span`, which keeps the planes inside
          the tile **and the nearest one on each side of it**, leaves ``pixel_frame_indices``
          **global**, and sets ``clip_start_frame`` to the tile's first pixel. DiffVAE then
          uses ``t_s(48) - t_s(56)`` for the 8-frame gap instead of rewriting 48 to -8.
        """
        my_tiles = dist_rank_tiles(self.vae_tiling, latent_shape, scale, self.rank, self.world_size)
        if self.vae_tiling.frames.num_tiles > 1 and tiling_config is not None:
            for tile in my_tiles:
                latent_f = tile.in_coords[0].stop - tile.in_coords[0].start
                pixel_f = (latent_f - 1) * scale.time + 1
                if _sgpu_has_temporal_tiling(tiling_config, pixel_f):
                    raise ValueError(
                        "Cannot combine multi-GPU temporal tiling (vae_tiling.frames.num_tiles > 1) "
                        "with single-GPU temporal tiling that would split this rank's tile "
                        f"({pixel_f} frames → {tiling_config.video_chunks_number(pixel_f)} chunks). "
                        "Use only one to avoid causal decoding conflicts."
                    )
        decoded = []
        for tile in my_tiles:
            # One MGPU tile only — cat stitches SGPU temporal yields of this tile,
            # not the full video (full assembly is gather_frames on the driver).
            latent_slice = latent[:, :, tile.in_coords[0], tile.in_coords[1], tile.in_coords[2]]
            tile_keyframes = (
                None
                if keyframes is None
                else keyframes.for_frame_span(tile.out_coords[0].start, tile.out_coords[0].stop - 1).crop_spatial(
                    tile.in_coords[1], tile.in_coords[2]
                )
            )
            # Only pass the keyword when there is something to pass: the wrapped decoder may be
            # any VideoDecoder, including one written against the pre-keyframes signature.
            extra = {} if tile_keyframes is None else {"keyframes": tile_keyframes}
            chunks = list(self.decoder.decode_video(latent_slice, tiling_config, generator=generator, **extra))
            pixels = torch.cat(chunks, dim=0)  # [F, H, W, C] in [0, 1]
            decoded.append(_to_decoded_tile(pixels, tile))
        return decoded

    def _collect_tiles(self, driver_tiles: list[DecodedTile]) -> list[DecodedTile]:
        """Collect tiles from all workers via the queue. Returns flat list of all tiles.
        Sorted by rank so the downstream reduction in ``gather_frames``
        (in-place ``+=`` over overlapping pixel regions) processes tiles in a
        fixed order. Queue-arrival order would otherwise vary run-to-run and
        yield 1-ulp bf16 drift from non-associative floating-point summation.
        """
        per_rank: dict[int, list[DecodedTile]] = {self.driver_rank: driver_tiles}
        for _ in range(self.world_size - 1):
            worker_rank, worker_tiles = self.queue.get()
            per_rank[worker_rank] = worker_tiles
        result: list[DecodedTile] = []
        for rank in sorted(per_rank):
            result.extend(per_rank[rank])
        return result
