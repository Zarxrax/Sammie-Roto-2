"""OpenImageIO push-pull fill engine."""
from object_removal.registry import EngineSpec
from object_removal.oiio_fill.engine import OIIOFillRemovalManager
from object_removal.oiio_fill.settings_ui import create_settings, load_settings, connect_preview


ENGINES = (EngineSpec(
    "OIIO Fill", "OIIO Fill", OIIOFillRemovalManager,
    create_settings, load_settings, connect_preview,
    "OpenImageIO push-pull fill for small or simple regions.", order=20,
),)
