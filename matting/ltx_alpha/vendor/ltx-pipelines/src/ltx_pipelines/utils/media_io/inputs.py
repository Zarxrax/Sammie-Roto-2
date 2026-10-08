"""Ingest handles for HDR IC-LoRA: MP4/MOV vs an EXR-frame folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ltx_pipelines.utils.media_io.color_config import EXRColorSpace


@dataclass(frozen=True, slots=True)
class VideoInput:
    """MP4/MOV source. ``gamma_encoded`` is display-referred sRGB (IEC 61966-2-1 EOTF)."""

    path: Path
    gamma_encoded: bool


@dataclass(frozen=True, slots=True)
class EXRVideoInput:
    """EXR-frame folder with a declared plate colour space and playback fps."""

    dir: Path
    color_space: EXRColorSpace
    frame_rate: float
