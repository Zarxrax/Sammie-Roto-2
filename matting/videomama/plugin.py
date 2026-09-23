"""VideoMaMa registration."""

from matting.registry import EngineSpec
from matting.videomama.engine import VideoMaMaManager
from matting.videomama.settings_ui import VideoMaMaSettings, save_defaults


INSTRUCTIONS = """
• Matting can be used to create mattes for objects with soft or poorly defined edges.<br>
• <b>Add points and run tracking in the Segmentation tab</b> so a mask is available on every frame, then press Run Matting.<br>
• VideoMaMa requires at least 8 GB of GPU memory.<br>
• VideoMaMa processes frames in batches. There may be temporal instability at batch boundaries.<br>
• VideoMaMa is free for non-commercial use and has
  <a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/blob/main/LICENSE.md">limited commercial use</a>.<br>
"""


ENGINES = (
    EngineSpec(
        id="VideoMaMa",
        label="VideoMaMa",
        manager_factory=VideoMaMaManager,
        instructions_html=INSTRUCTIONS,
        settings_widget_factory=VideoMaMaSettings,
        unsupported_device_types=("cpu",),
        order=30,
        save_defaults=save_defaults,
    ),
)
