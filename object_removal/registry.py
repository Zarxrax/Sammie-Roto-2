"""Discover object removal engines from plugin folders."""
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pkgutil import iter_modules
from typing import Callable

import object_removal

DEFAULT_ENGINE_ID = "MiniMax-Remover"


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    manager_factory: Callable
    settings_factory: Callable
    load_settings: Callable
    connect_preview: Callable
    hint: str = ""
    unsupported_device_types: tuple[str, ...] = ()
    offload_segmentation: bool = False
    order: int = 100
    save_defaults: Callable | None = None


def get_engine_specs():
    specs = []
    for module in iter_modules(object_removal.__path__):
        if module.ispkg and find_spec(f"object_removal.{module.name}.plugin"):
            specs.extend(import_module(f"object_removal.{module.name}.plugin").ENGINES)
    ids = [spec.id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Object removal engine IDs must be unique")
    return sorted(specs, key=lambda spec: (spec.order, spec.label))


def get_engine(engine_id):
    return next((spec for spec in get_engine_specs() if spec.id == engine_id), None)


def create_removal_manager(engine_id=None):
    if engine_id is None:
        from sammie.settings_manager import get_settings_manager
        engine_id = get_settings_manager().get_session_setting("removal_method", DEFAULT_ENGINE_ID)
    spec = get_engine(engine_id) or get_engine(DEFAULT_ENGINE_ID)
    if spec is None:
        raise ValueError("No object removal engines are available")
    return spec.manager_factory()
