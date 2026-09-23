"""Discover matting engines from folders containing a plugin.py module.

Each plugin exports ENGINES, a sequence of EngineSpec objects. Adding an engine
folder requires no changes to the application window or this registry.
"""

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pkgutil import iter_modules
from typing import Callable

import matting
from sammie.settings_manager import get_settings_manager


DEFAULT_ENGINE_ID = "MatAnyone2"


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    manager_factory: Callable
    instructions_html: str
    settings_widget_factory: Callable | None = None
    unsupported_device_types: tuple[str, ...] = ()
    order: int = 100
    save_defaults: Callable | None = None


def get_engine_specs():
    """Return all registered engines in UI order."""
    specs = []
    for module in iter_modules(matting.__path__):
        if not module.ispkg:
            continue
        plugin_name = f"matting.{module.name}.plugin"
        if find_spec(plugin_name) is None:
            continue
        plugin = import_module(plugin_name)
        specs.extend(plugin.ENGINES)
    ids = [spec.id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Matting engine IDs must be unique")
    return sorted(specs, key=lambda spec: (spec.order, spec.label))


def get_engine(engine_id):
    return next((spec for spec in get_engine_specs() if spec.id == engine_id), None)


def create_matting_manager(engine_id=None):
    if engine_id is None:
        engine_id = get_settings_manager().get_session_setting("matany_model", DEFAULT_ENGINE_ID)
    spec = get_engine(engine_id)
    if spec is None:
        spec = get_engine(DEFAULT_ENGINE_ID)
    if spec is None:
        raise ValueError("No matting engines are available")
    return spec.manager_factory()
