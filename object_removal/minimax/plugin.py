"""MiniMax video diffusion removal engine."""
from object_removal.registry import EngineSpec
from object_removal.minimax.engine import MinimaxRemovalManager
from object_removal.minimax.settings_ui import create_settings, load_settings, connect_preview


def save_defaults(settings_mgr):
    for key, fallback in (("minimax_resolution", 480), ("minimax_vae_tiling", False), ("minimax_steps", 6)):
        settings_mgr.set_app_setting(f"default_{key}", settings_mgr.get_session_setting(key, fallback))


ENGINES = (EngineSpec(
    "MiniMax-Remover", "MiniMax-Remover", MinimaxRemovalManager,
    create_settings, load_settings, connect_preview,
    "Video diffusion removal; memory use grows with clip length.",
    unsupported_device_types=("cpu",), offload_segmentation=True,
    order=10, save_defaults=save_defaults,
),)
