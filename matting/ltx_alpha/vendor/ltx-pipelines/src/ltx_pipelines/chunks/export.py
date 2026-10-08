"""Chunk stream I/O: overlap-aware reference decode, lazy stitch, and ``PipelineOutput`` packing."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Generator, Iterator
from itertools import chain

import torch
from torch._prims_common import DeviceLikeType

from ltx_core.color.audio_mux import validate_audio_waveform
from ltx_core.model.video_vae import TilingConfig
from ltx_core.types import Audio
from ltx_pipelines.chunks.audio import deferred_stitch_audio, stitch_audio
from ltx_pipelines.chunks.state import DecodedChunk
from ltx_pipelines.utils.media_io import decode_video_by_frame
from ltx_pipelines.utils.types import PipelineOutput

logger = logging.getLogger(__name__)


class SequentialVideoFrameSource:
    """Decode a video once, in order, reusing overlap frames already yielded.
    ``take(start, count)`` yields frames ``[start, start+count)``. Later windows may
    overlap; kept frames stay until a later ``take`` starts past them. Asking for a
    frame that has already been dropped raises.
    """

    def __init__(self, path: str, device: DeviceLikeType) -> None:
        self._path = path
        self._device = device
        self._gen: Generator[torch.Tensor, None, None] | None = None
        self._next = 0
        self._kept_start = 0
        self._kept: deque[torch.Tensor] = deque()

    def take(self, start: int, count: int) -> Iterator[torch.Tensor]:
        if start < 0 or count < 0:
            raise ValueError(f"start and count must be >= 0, got start={start} count={count}")
        end = start + count
        if start < self._kept_start:
            raise ValueError(f"cannot take frames [{start}:{end}); lookback starts at {self._kept_start}")
        if start > self._next:
            raise ValueError(f"cannot skip from frame {self._next} to {start}; take windows in order")
        while self._kept_start < start:
            self._kept.popleft()
            self._kept_start += 1
        if end > self._next:
            if self._gen is None:
                self._gen = decode_video_by_frame(self._path, self._device, starting_frame=self._next)
            while self._next < end:
                try:
                    self._kept.append(next(self._gen))
                except StopIteration:
                    break
                self._next += 1
        return iter(list(self._kept)[:count])

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()
            self._gen = None


def split_decoded_chunks(
    chunks: Iterator[DecodedChunk],
) -> tuple[Iterator[torch.Tensor], Iterator[Audio] | None, int | None]:
    """Split decoded chunks into a video iterator, stitched audio, and sample rate.
    Drain ``video`` before iterating ``audio``; audio is filled as video is
    consumed. Both audio fields are ``None`` when the stream is video-only.
    """
    first = next(chunks, None)
    if first is None:
        raise ValueError("split_decoded_chunks received no chunks")

    height, width = first.video.shape[-3], first.video.shape[-2]
    has_audio = first.audio is not None
    sample_rate = first.audio.sampling_rate if has_audio else None
    audio_chunks: list[Audio] = []

    def validate_and_collect(chunk: DecodedChunk) -> None:
        if (chunk.audio is not None) != has_audio:
            raise ValueError("audio presence must be the same for every chunk")
        if chunk.video.shape[0] == 0:
            raise ValueError("chunk had no video frames")
        chunk_h, chunk_w = chunk.video.shape[-3], chunk.video.shape[-2]
        if (chunk_h, chunk_w) != (height, width):
            raise ValueError(f"chunk video size {chunk_w}x{chunk_h} does not match expected {width}x{height}")
        if chunk.audio is not None:
            validate_audio_waveform(chunk.audio)
            audio_chunks.append(chunk.audio)

    def video() -> Iterator[torch.Tensor]:
        for chunk in chain((first,), chunks):
            validate_and_collect(chunk)
            yield chunk.video

    audio = None if sample_rate is None else deferred_stitch_audio(audio_chunks)
    return video(), audio, sample_rate


def pipeline_output_from_chunks(
    chunks: Iterator[DecodedChunk],
    *,
    num_frames: int,
    tiling_config: TilingConfig | None,
) -> PipelineOutput:
    """Pack decoded chunks into :class:`PipelineOutput` for ``__call__``.
    One chunk keeps that chunk's video/audio identity. Several chunks yield
    each video tensor in order and one concatenated audio track.
    """
    decoded = list(chunks)
    if not decoded:
        raise ValueError("expected a decoded chunk")
    audios: list[Audio] = []
    for chunk in decoded:
        if chunk.audio is None:
            raise ValueError("expected decoded audio")
        validate_audio_waveform(chunk.audio)
        audios.append(chunk.audio)
    if len(decoded) == 1:
        only = decoded[0]
        return PipelineOutput(iter((only.video,)), audios[0], num_frames, tiling_config)
    stitched = stitch_audio(audios)
    assert stitched is not None
    return PipelineOutput((chunk.video for chunk in decoded), stitched, num_frames, tiling_config)
