"""Matting engine plugins and shared infrastructure."""

from matting.registry import (
    DEFAULT_ENGINE_ID, EngineSpec, create_matting_manager, get_engine,
    get_engine_specs,
)
