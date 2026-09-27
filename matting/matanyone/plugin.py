"""MatAnyone and MatAnyone2 registrations."""

from matting.registry import EngineSpec
from matting.matanyone.engine import MatAnyManager


INSTRUCTIONS = """
• Matting can be used to create mattes for objects with soft or poorly defined edges.<br>
• <b>Add points to at least one frame in the Segmentation tab</b>, then press Run Matting.<br>
• The MatAnyone models are faster and require less VRAM than VideoMaMa, but may be less accurate.<br>
• If you add points to multiple frames, matting refreshes at each keyframe, which may momentarily affect temporal stability.<br>
• MatAnyone is free for non-commercial use and requires
  <a href="https://github.com/pq-yang/MatAnyone?tab=License-1-ov-file">permission for commercial use</a>.<br>
"""


ENGINES = (
    EngineSpec("MatAnyone", "MatAnyone", lambda: MatAnyManager("MatAnyone"),
               INSTRUCTIONS, order=10),
    EngineSpec("MatAnyone2", "MatAnyone2", lambda: MatAnyManager("MatAnyone2"),
               INSTRUCTIONS, order=20),
)
