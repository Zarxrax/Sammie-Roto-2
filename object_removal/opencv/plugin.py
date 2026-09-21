"""OpenCV frame inpainting engine."""
from object_removal.registry import EngineSpec
from object_removal.opencv.engine import OpenCVRemovalManager
from object_removal.opencv.settings_ui import create_settings, load_settings, connect_preview


def save_defaults(settings_mgr):
    for key, fallback in (("inpaint_method", "Telea"), ("inpaint_radius", 3)):
        settings_mgr.set_app_setting(f"default_{key}", settings_mgr.get_session_setting(key, fallback))


ENGINES = (EngineSpec(
    "OpenCV", "OpenCV", OpenCVRemovalManager,
    create_settings, load_settings, connect_preview,
    "Fast traditional inpainting with limited quality.",
    order=20, save_defaults=save_defaults,
),)
