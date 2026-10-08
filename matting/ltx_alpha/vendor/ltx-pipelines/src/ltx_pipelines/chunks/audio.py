"""Concatenate chunk audio parts after the video iterator has been drained."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from ltx_core.color.audio_mux import normalize_audio_waveform
from ltx_core.types import Audio


def stitch_audio(parts: list[Audio]) -> Audio | None:
    """Concatenate chunk parts into one ``Audio``, or ``None`` when empty.
    Parts must already be valid stereo. Call only after every part is available
    (e.g. after the video iterator has been consumed).
    """
    if not parts:
        return None
    sample_rate = parts[0].sampling_rate
    tensors: list[torch.Tensor] = []
    for audio in parts:
        if audio.sampling_rate != sample_rate:
            raise ValueError(f"chunk audio sampling rate {audio.sampling_rate} does not match expected {sample_rate}")
        tensors.append(normalize_audio_waveform(audio.waveform).cpu())
    return Audio(waveform=torch.cat(tensors, dim=0), sampling_rate=sample_rate)


def deferred_stitch_audio(parts: list[Audio]) -> Iterator[Audio]:
    """Lazy iterator that stitches ``parts`` after video drain.
    ``split_decoded_chunks`` appends to ``parts`` while video is consumed; pull
    this iterator only once every part is present.
    """
    stitched = stitch_audio(parts)
    if stitched is not None:
        yield stitched
