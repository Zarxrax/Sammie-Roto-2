from PySide6.QtWidgets import QComboBox, QFormLayout, QSpinBox, QWidget

from sammie.settings_manager import get_settings_manager


class LTXAlphaSettings(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)
        settings = get_settings_manager()

        self.chunk_frames = QComboBox()
        self.chunk_frames.addItems(["17", "25", "49", "73", "97", "145"])
        self.chunk_frames.setToolTip(
            "Frames generated together. Larger chunks improve temporal continuity but use more memory. "
            "LTX requires an 8n+1 frame count."
        )
        self.chunk_frames.currentTextChanged.connect(
            lambda value: settings.set_session_setting("ltx_alpha_chunk_frames", int(value)))
        layout.addRow("Frames per chunk:", self.chunk_frames)

        self.crop_mode = QComboBox()
        self.crop_mode.addItem("Global crop", "global")
        self.crop_mode.addItem("Chunk crop", "chunk")
        self.crop_mode.setToolTip(
            "Uses existing segmentation only to reduce the RGB processing area. "
            "Global crop uses one region for the complete range. Chunk crop finds a smaller region per chunk. "
            "If segmentation is unavailable, the full frame is processed."
        )
        self.crop_mode.currentIndexChanged.connect(
            lambda _index: settings.set_session_setting(
                "ltx_alpha_crop_mode", self.crop_mode.currentData()))
        layout.addRow("Segmentation crop:", self.crop_mode)
        self.set_segmentation_available(False)

        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setToolTip("Generation seed. The same seed and input produce repeatable results.")
        self.seed.valueChanged.connect(
            lambda value: settings.set_session_setting("ltx_alpha_seed", value))
        layout.addRow("Seed:", self.seed)

        self.load_settings()

    def load_settings(self):
        settings = get_settings_manager()
        self.chunk_frames.setCurrentText(str(settings.get_session_setting("ltx_alpha_chunk_frames", 49)))
        crop_index = self.crop_mode.findData(
            settings.get_session_setting("ltx_alpha_crop_mode", "chunk"))
        self.crop_mode.setCurrentIndex(max(0, crop_index))
        self.seed.setValue(settings.get_session_setting("ltx_alpha_seed", 1234))

    def set_segmentation_available(self, available):
        self.crop_mode.setEnabled(available)
        if available:
            self.crop_mode.setToolTip(
                "Uses existing segmentation only to reduce the RGB processing area. "
                "Global crop uses one region for the complete range. Chunk crop finds a smaller region per chunk."
            )
        else:
            self.crop_mode.setToolTip(
                "Create or load segmentation masks to enable global and chunk cropping. "
                "LTX Alpha will process the full frame without segmentation."
            )


def save_defaults(settings):
    for key, fallback in (
        ("ltx_alpha_chunk_frames", 49),
        ("ltx_alpha_crop_mode", "chunk"),
        ("ltx_alpha_seed", 1234),
    ):
        settings.set_app_setting(f"default_{key}", settings.get_session_setting(key, fallback))
