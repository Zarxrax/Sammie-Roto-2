"""VideoMaMa-specific processing controls."""

from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QSpinBox, QWidget

from sammie.settings_manager import get_settings_manager


class VideoMaMaSettings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        settings = get_settings_manager()
        layout = QFormLayout(self)

        self.overlap = QComboBox()
        self.overlap.addItems(["0", "2", "4"])
        self.overlap.setToolTip("Overlapping frames between batches. Higher values can smooth batch boundaries.")
        self.overlap.currentTextChanged.connect(
            lambda value: settings.set_session_setting("matany_overlap", int(value)))
        layout.addRow("Crossfade overlap frames:", self.overlap)

        self.batch_size = QComboBox()
        self.batch_size.addItems(["16", "32", "64", "128", "256", "512"])
        self.batch_size.setToolTip("Frames processed together. Higher values require more GPU memory.")
        self.batch_size.currentTextChanged.connect(
            lambda value: settings.set_session_setting("matany_chunk", int(value)))
        layout.addRow("Frames per batch:", self.batch_size)

        self.memory_enabled = QCheckBox("Memory budget")
        self.memory_enabled.setToolTip(
            "Split the cropped region into overlapping tiles and limit PyTorch GPU allocations "
            "on MPS or CUDA. The general MPS safety ceiling remains active when disabled.")
        self.memory_enabled.toggled.connect(
            lambda value: settings.set_session_setting("videomama_memory_enabled", value))
        layout.addRow(self.memory_enabled)

        self.memory_gib = QSpinBox()
        self.memory_gib.setRange(8, 80)
        self.memory_gib.setSuffix(" GiB")
        self.memory_gib.setToolTip(
            "Lower values create smaller tiles and use less GPU memory, but take longer "
            "and may show tile seams. This does not cap total system RAM.")
        self.memory_gib.valueChanged.connect(
            lambda value: settings.set_session_setting("videomama_memory_gib", value))
        self.memory_enabled.toggled.connect(self.memory_gib.setEnabled)
        layout.addRow("GPU allocation target:", self.memory_gib)

    def load_settings(self):
        settings = get_settings_manager()
        self.overlap.setCurrentText(str(settings.get_session_setting("matany_overlap", 2)))
        batch_size = max(16, settings.get_session_setting("matany_chunk", 16))
        settings.set_session_setting("matany_chunk", batch_size)
        self.batch_size.setCurrentText(str(batch_size))
        self.memory_gib.setValue(settings.get_session_setting("videomama_memory_gib", 32))
        self.memory_enabled.setChecked(
            settings.get_session_setting("videomama_memory_enabled", True))
        self.memory_gib.setEnabled(self.memory_enabled.isChecked())


def save_defaults(settings):
    for key, fallback in (
        ("matany_overlap", 2), ("matany_chunk", 16),
        ("videomama_memory_gib", 32), ("videomama_memory_enabled", True),
    ):
        settings.set_app_setting(f"default_{key}", settings.get_session_setting(key, fallback))
