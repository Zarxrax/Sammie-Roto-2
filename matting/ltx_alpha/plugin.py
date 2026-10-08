"""LTX 2.5 Alpha Gen registration."""

from matting.ltx_alpha.engine import LTXAlphaManager
from matting.ltx_alpha.settings_ui import LTXAlphaSettings, save_defaults
from matting.registry import EngineSpec


INSTRUCTIONS = """
• Generates a full-frame alpha matte directly from the RGB video.<br>
• While segmentation is not needed, it can be used to reduce memory use and speed up matting.<br>
• The model chooses the foreground; individual objects cannot be selected.<br>
• Existing segmentation can define one global crop or a separate maximum crop for each chunk; it is not passed to the model.<br>
• Clips are processed in consecutive chunks of at most 145 frames.<br>
• <b>This is a VRAM-hungry matting process. You need to find the chunk × resolution balance that fits your available VRAM.</b><br>
• The transformer quantization is selected during download; lower quants use less memory but may reduce matte quality.<br>
• The Alpha Gen adapter requires access to its gated Hugging Face repository.<br>
• If you experience alpha weirdness or results showing RGB instead of alpha: try changing the internal resolution or a better (bigger) GGUF quantization.<br>
• Uses the LTX 2.x Community License. Review its terms before use.<br>
"""


ENGINES = (
    EngineSpec(
        id="LTXAlpha",
        label="LTX 2.5 Alpha Gen",
        manager_factory=LTXAlphaManager,
        instructions_html=INSTRUCTIONS,
        settings_widget_factory=LTXAlphaSettings,
        unsupported_device_types=("cpu",),
        order=40,
        save_defaults=save_defaults,
        requires_segmentation=False,
    ),
)
