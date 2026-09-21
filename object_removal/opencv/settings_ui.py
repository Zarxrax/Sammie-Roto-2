from PySide6.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QHBoxLayout, QLabel, QComboBox, QGridLayout, QSlider, QCheckBox
from PySide6.QtCore import Qt
from sammie.settings_manager import get_settings_manager
from sammie.gui_widgets import ClickableLabel

def create_settings(tab):
    widget = QWidget()
    layout = QVBoxLayout(widget)
    """Create parameters for OpenCV method"""
    settings_mgr = get_settings_manager()

    tab.opencv_params_group = QWidget()
    opencv_layout = QVBoxLayout(tab.opencv_params_group)
    opencv_layout.setContentsMargins(0, 0, 0, 0)

    # Algorithm selection
    algorithm_group = QGroupBox("Algorithm")
    algorithm_layout = QHBoxLayout(algorithm_group)

    algorithm_layout.addWidget(QLabel("Algorithm:"))

    tab.opencv_algorithm_combo = QComboBox()
    tab.opencv_algorithm_combo.addItems(["Telea", "Navier-Stokes"])

    current_algorithm = settings_mgr.get_session_setting("inpaint_method", "Telea")
    index = tab.opencv_algorithm_combo.findText(current_algorithm)
    if index >= 0:
        tab.opencv_algorithm_combo.setCurrentIndex(index)

    tab.opencv_algorithm_combo.setToolTip("Telea: Based on fast marching method.\nNavier-Stokes: Fluid dynamics based method, may produce smoother results.")
    tab.opencv_algorithm_combo.currentTextChanged.connect(
        lambda algorithm: settings_mgr.set_session_setting("inpaint_method", algorithm)
    )

    algorithm_layout.addWidget(tab.opencv_algorithm_combo)
    algorithm_layout.addStretch()

    opencv_layout.addWidget(algorithm_group)

    # OpenCV-specific sliders
    sliders_group = QGroupBox("Parameters")
    sliders_layout = QGridLayout(sliders_group)

    slider_configs = [
        ("Inpaint Radius:", 1, 10, "inpaint_radius", 3,
        "The radius of a circular neighborhood of each point inpainted that is considered by the algorithm.",
        lambda v: str(v), lambda v: v, lambda v: v)
    ]

    for i, (label_text, min_val, max_val, setting_key, fallback_default, tooltip,
            display_func, slider_func, save_func) in enumerate(slider_configs):

        default_val = getattr(settings_mgr.app_settings, f"default_{setting_key}", fallback_default)
        current_val = settings_mgr.get_session_setting(setting_key, default_val)

        label = ClickableLabel(label_text)
        label.setToolTip(f"Double-click to reset to default value ({display_func(slider_func(default_val))})")
        sliders_layout.addWidget(label, i, 0)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(slider_func(current_val))
        slider.setToolTip(tooltip)
        sliders_layout.addWidget(slider, i, 1)

        value_label = QLabel(display_func(slider_func(current_val)))
        value_label.setMinimumWidth(30)
        value_label.setAlignment(Qt.AlignCenter)
        sliders_layout.addWidget(value_label, i, 2)

        slider.valueChanged.connect(
            lambda v, lbl=value_label, func=display_func: lbl.setText(func(v))
        )
        slider.valueChanged.connect(
                lambda v, key=setting_key, func=save_func: settings_mgr.set_session_setting(key, func(v))
        )

        label.doubleClicked.connect(
            lambda s=slider, default=default_val, func=slider_func: tab._reset_slider_to_default(s, func(default))
        )

        if setting_key == "inpaint_radius":
            tab.opencv_radius_slider = slider
            tab.opencv_radius_value = value_label

    opencv_layout.addWidget(sliders_group)
    layout.addWidget(tab.opencv_params_group)

    return widget

def load_settings(tab):
    settings_mgr = get_settings_manager()
    tab.opencv_algorithm_combo.setCurrentText(settings_mgr.get_session_setting("inpaint_method", "Telea"))
    tab.opencv_radius_slider.setValue(settings_mgr.get_session_setting("inpaint_radius", 3))


def connect_preview(tab, callback):
    tab.opencv_algorithm_combo.currentTextChanged.connect(callback)
    tab.opencv_radius_slider.valueChanged.connect(callback)
