from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QComboBox, QGridLayout, QSlider, QCheckBox
from PySide6.QtCore import Qt
from sammie.settings_manager import get_settings_manager
from sammie.gui_widgets import ClickableLabel

def create_settings(tab):
    widget = QWidget()
    layout = QVBoxLayout(widget)
    """Create parameters for MiniMax-Remover method"""
    settings_mgr = get_settings_manager()

    tab.minimax_params_group = QWidget()
    minimax_layout = QVBoxLayout(tab.minimax_params_group)
    minimax_layout.setContentsMargins(0, 0, 0, 0)

    params_group = QGroupBox("Parameters")
    params_layout = QGridLayout(params_group)

    row = 0

    # Internal Resolution
    params_layout.addWidget(QLabel("Internal Resolution:"), row, 0)
    tab.minimax_resolution_combo = QComboBox()
    tab.minimax_resolution_combo.addItems(["352", "480", "720", "1080"])

    current_resolution = str(settings_mgr.get_session_setting("minimax_resolution", 480))
    index = tab.minimax_resolution_combo.findText(current_resolution)
    if index >= 0:
        tab.minimax_resolution_combo.setCurrentIndex(index)

    tab.minimax_resolution_combo.setToolTip("Internal processing resolution. Higher values produce better quality but are slower and use more VRAM.")
    tab.minimax_resolution_combo.currentTextChanged.connect(
        lambda v: settings_mgr.set_session_setting("minimax_resolution", int(v))
    )
    params_layout.addWidget(tab.minimax_resolution_combo, row, 1, 1, 2)

    row += 1

    # VAE Tiling checkbox
    params_layout.addWidget(QLabel("Use VAE Tiling:"), row, 0)
    tab.minimax_vae_tiling_checkbox = QCheckBox()

    vae_tiling = settings_mgr.get_session_setting("minimax_vae_tiling", False)
    tab.minimax_vae_tiling_checkbox.setChecked(vae_tiling)
    tab.minimax_vae_tiling_checkbox.setToolTip("If you get an out of memory error during the VAE decode step, try enabling this option. The VAE steps will take longer but use less VRAM.")
    tab.minimax_vae_tiling_checkbox.stateChanged.connect(
        lambda state: settings_mgr.set_session_setting("minimax_vae_tiling", tab.minimax_vae_tiling_checkbox.isChecked())
    )
    params_layout.addWidget(tab.minimax_vae_tiling_checkbox, row, 1, 1, 2)

    row += 1

    # Steps slider
    default_steps = getattr(settings_mgr.app_settings, "default_minimax_steps", 6)
    current_steps = settings_mgr.get_session_setting("minimax_steps", default_steps)

    label = ClickableLabel("Steps:")
    label.setToolTip(f"Double-click to reset to default value ({default_steps})")
    params_layout.addWidget(label, row, 0)

    tab.minimax_steps_slider = QSlider(Qt.Horizontal)
    tab.minimax_steps_slider.setRange(4, 12)
    tab.minimax_steps_slider.setValue(current_steps)
    tab.minimax_steps_slider.setToolTip("Number of diffusion steps. Larger values are better quality but slower.")
    params_layout.addWidget(tab.minimax_steps_slider, row, 1)

    tab.minimax_steps_value = QLabel(str(current_steps))
    tab.minimax_steps_value.setMinimumWidth(30)
    tab.minimax_steps_value.setAlignment(Qt.AlignCenter)
    params_layout.addWidget(tab.minimax_steps_value, row, 2)

    tab.minimax_steps_slider.valueChanged.connect(
        lambda v: tab.minimax_steps_value.setText(str(v))
    )
    tab.minimax_steps_slider.valueChanged.connect(
        lambda v: settings_mgr.set_session_setting("minimax_steps", v)
    )

    label.doubleClicked.connect(
        lambda: tab._reset_slider_to_default(tab.minimax_steps_slider, default_steps)
    )

    minimax_layout.addWidget(params_group)
    layout.addWidget(tab.minimax_params_group)

    return widget

def load_settings(tab):
    settings_mgr = get_settings_manager()
    tab.minimax_resolution_combo.setCurrentText(str(settings_mgr.get_session_setting("minimax_resolution", 480)))
    tab.minimax_vae_tiling_checkbox.setChecked(settings_mgr.get_session_setting("minimax_vae_tiling", False))
    tab.minimax_steps_slider.setValue(settings_mgr.get_session_setting("minimax_steps", 6))


def connect_preview(tab, callback):
    tab.minimax_resolution_combo.currentTextChanged.connect(callback)
    tab.minimax_vae_tiling_checkbox.stateChanged.connect(callback)
    tab.minimax_steps_slider.valueChanged.connect(callback)
