"""Compatibility imports for matting engine plugins."""

from matting.base import MattingManager
from matting.registry import (
    DEFAULT_ENGINE_ID, EngineSpec, create_matting_manager, get_engine,
    get_engine_specs,
)
