"""Discover segmentation engines from plugin folders."""
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pkgutil import iter_modules
from typing import Callable

import segmentation

DEFAULT_ENGINE_ID = "Base"


@dataclass(frozen=True)
class EngineSpec:
    id: str
    label: str
    load_predictor: Callable
    hint: str = ""
    order: int = 100


def get_engine_specs():
    specs = []
    for module in iter_modules(segmentation.__path__):
        if module.ispkg and find_spec(f"segmentation.{module.name}.plugin"):
            specs.extend(import_module(f"segmentation.{module.name}.plugin").ENGINES)
    ids = [spec.id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Segmentation engine IDs must be unique")
    return sorted(specs, key=lambda spec: (spec.order, spec.label))


def get_engine(engine_id):
    return next((spec for spec in get_engine_specs() if spec.id == engine_id), None)
