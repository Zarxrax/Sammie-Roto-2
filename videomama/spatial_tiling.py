"""Memory-budgeted spatial tiling for VideoMaMa inference."""

from dataclasses import dataclass
from math import ceil, sqrt

import numpy as np


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0


def _axis_tiles(length, max_side, overlap):
    """Return aligned spans that cover an axis with approximately even overlap."""
    if length <= max_side:
        return [(0, length)]
    count = ceil((length - overlap) / (max_side - overlap))
    side = ceil((length + (count - 1) * overlap) / count / 8) * 8
    starts = [round(i * (length - side) / (count - 1) / 8) * 8 for i in range(count)]
    return [(start, start + side) for start in starts]


def plan_tiles(width, height, frames, memory_gib, overlap=64):
    """Plan tiles from the observed 4-frame, 984x1008, 52 GiB MPS run.

    The conservative estimate models transient attention memory as proportional
    to frame count times spatial area squared. The allocator limit is the hard
    safety bound; this estimate only chooses a starting tile size.
    """
    if width % 8 or height % 8:
        raise ValueError("VideoMaMa tile dimensions must be divisible by 8")
    if frames < 1 or memory_gib < 8:
        raise ValueError("VideoMaMa needs at least one frame and an 8 GiB memory budget")

    reference_area = 984 * 1008
    fixed_gib = 4.0
    reference_transient_gib = 52.0 - fixed_gib
    area_ratio = sqrt(((memory_gib - fixed_gib) / reference_transient_gib) * (4 / frames))
    max_pixels = reference_area * area_ratio * 0.75
    max_side = max(128, int(sqrt(max_pixels) // 8) * 8)
    overlap = min(overlap, max_side // 4)
    overlap = max(8, overlap // 8 * 8)

    xs = _axis_tiles(width, max_side, overlap)
    ys = _axis_tiles(height, max_side, overlap)
    tiles = []
    for row, (y0, y1) in enumerate(ys):
        for col, (x0, x1) in enumerate(xs):
            tiles.append(Tile(
                x0, y0, x1, y1,
                left=xs[col - 1][1] - x0 if col else 0,
                right=x1 - xs[col + 1][0] if col + 1 < len(xs) else 0,
                top=ys[row - 1][1] - y0 if row else 0,
                bottom=y1 - ys[row + 1][0] if row + 1 < len(ys) else 0,
            ))
    return tiles


def _blend_weights(tile):
    height = tile.y1 - tile.y0
    width = tile.x1 - tile.x0
    x = np.ones(width, dtype=np.float32)
    y = np.ones(height, dtype=np.float32)
    if tile.left:
        x[:tile.left] *= np.linspace(0, 1, tile.left, endpoint=False, dtype=np.float32)
    if tile.right:
        x[-tile.right:] *= np.linspace(1, 0, tile.right, endpoint=False, dtype=np.float32)
    if tile.top:
        y[:tile.top] *= np.linspace(0, 1, tile.top, endpoint=False, dtype=np.float32)
    if tile.bottom:
        y[-tile.bottom:] *= np.linspace(1, 0, tile.bottom, endpoint=False, dtype=np.float32)
    return y[:, None] * x[None, :]


def run_tiled(pipeline, cond_frames, mask_frames, memory_gib, seed=42,
              progress_callback=None):
    """Run each spatial tile independently and feather its output into a matte."""
    height, width = cond_frames[0].shape[:2]
    frame_count = len(cond_frames)
    tiles = plan_tiles(width, height, frame_count, memory_gib)
    if len(tiles) == 1:
        return pipeline.run(cond_frames, mask_frames, seed=seed,
                            progress_callback=progress_callback)

    print(f"VideoMaMa: {width}x{height} crop, {frame_count} frames, "
          f"{memory_gib} GiB budget -> {len(tiles)} overlapping spatial tiles")

    accum = np.zeros((frame_count, height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    for index, tile in enumerate(tiles, 1):
        tile_size = f"{tile.x1 - tile.x0}x{tile.y1 - tile.y0}"
        if progress_callback is not None:
            progress_callback(index - 1, len(tiles), f"Tile {index}/{len(tiles)} ({tile_size})")

        def tile_progress(_step, _total, description):
            if progress_callback is not None:
                progress_callback(index, len(tiles),
                                  f"Tile {index}/{len(tiles)} ({tile_size}): {description}")

        cond_tile = [frame[tile.y0:tile.y1, tile.x0:tile.x1] for frame in cond_frames]
        mask_tile = [frame[tile.y0:tile.y1, tile.x0:tile.x1] for frame in mask_frames]
        # Align model inputs to 64 pixels. This avoids MPS resample layouts
        # encountered with arbitrary UNet feature-map sizes.
        pad_h = -(tile.y1 - tile.y0) % 64
        pad_w = -(tile.x1 - tile.x0) % 64
        if pad_h or pad_w:
            cond_tile = [np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)),
                                mode="edge") for frame in cond_tile]
            mask_tile = [np.pad(frame, ((0, pad_h), (0, pad_w)),
                                mode="edge") for frame in mask_tile]
        output = pipeline.run(cond_tile, mask_tile, seed=seed,
                              progress_callback=tile_progress)
        blend = _blend_weights(tile)
        weights[tile.y0:tile.y1, tile.x0:tile.x1] += blend
        for frame_index, frame in enumerate(output):
            accum[frame_index, tile.y0:tile.y1, tile.x0:tile.x1] += (
                frame[:tile.y1 - tile.y0, :tile.x1 - tile.x0, 0] * blend)
        del output

    if np.any(weights <= 0):
        raise RuntimeError("VideoMaMa tiles left pixels without blending weights")
    matte = np.clip(accum / weights[None, :, :], 0, 255).astype(np.uint8)
    return [np.repeat(frame[:, :, None], 3, axis=2) for frame in matte]
