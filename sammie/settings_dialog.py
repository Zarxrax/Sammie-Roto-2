# sammie/settings_dialog.py
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QGroupBox, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, 
    QPushButton, QComboBox, QSlider, QWidget, QFormLayout, QScrollArea,
    QDialogButtonBox
)
from PySide6.QtCore import Qt
from sammie.settings_manager import SettingsManager
from sammie.gui_widgets import (
    ColorPickerWidget, ColorDisplayWidget
)

class SettingsDialog(QDialog):
    """Settings dialog for configuring application and default session settings"""
    
    def __init__(self, settings_manager: SettingsManager, parent=None):
        super().__init__(parent)
        self.settings_mgr = settings_manager
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(450, 500)
        
        # Store original settings for cancel functionality
        self._original_app_settings = self._backup_app_settings()
        
        self._init_ui()
        self._load_current_values()
        
    def _backup_app_settings(self):
        """Create a backup of current application settings"""
        from sammie.settings_manager import ApplicationSettings
        import copy
        return copy.deepcopy(self.settings_mgr.app_settings)
    
    def _restore_app_settings(self):
        """Restore application settings from backup"""
        self.settings_mgr.app_settings = self._original_app_settings
    
    def _init_ui(self):
        """Initialize the settings dialog UI"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Add tabs with scroll areas
        general_tab = self._create_general_tab()
        defaults_tab = self._create_defaults_tab()
        
        # Wrap each tab in a scroll area
        general_scroll = QScrollArea()
        general_scroll.setWidget(general_tab)
        general_scroll.setWidgetResizable(True)
        general_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        defaults_scroll = QScrollArea()
        defaults_scroll.setWidget(defaults_tab)
        defaults_scroll.setWidgetResizable(True)
        defaults_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.tab_widget.addTab(general_scroll, "General")
        self.tab_widget.addTab(defaults_scroll, "Defaults")
        
        layout.addWidget(self.tab_widget)
        
        # Buttons using QDialogButtonBox for platform-specific ordering
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout.addWidget(self.button_box)
        
    def _create_defaults_tab(self):
        """Create the defaults settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # View defaults group
        view_group = QGroupBox("View Defaults")
        view_layout = QFormLayout(view_group)
        
        self.default_show_masks_cb = QCheckBox()
        view_layout.addRow("Show Masks by Default:", self.default_show_masks_cb)
        
        self.default_show_outlines_cb = QCheckBox()
        view_layout.addRow("Show Outlines by Default:", self.default_show_outlines_cb)
        
        self.default_antialias_cb = QCheckBox()
        view_layout.addRow("Antialias by Default:", self.default_antialias_cb)

        self.default_color_picker = ColorPickerWidget()
        view_layout.addRow("Default Background Color:", self.default_color_picker)
        
        layout.addWidget(view_group)
        
        # Processing defaults group
        proc_group = QGroupBox("Segmentation Postprocessing Defaults")
        proc_layout = QFormLayout(proc_group)
        
        self.default_holes_spin = QSpinBox()
        self.default_holes_spin.setRange(0, 50)
        proc_layout.addRow("Default Remove Holes:", self.default_holes_spin)
        
        self.default_dots_spin = QSpinBox()
        self.default_dots_spin.setRange(0, 50)
        proc_layout.addRow("Default Remove Dots:", self.default_dots_spin)
        
        self.default_border_spin = QSpinBox()
        self.default_border_spin.setRange(0, 10)
        proc_layout.addRow("Default Border Fix:", self.default_border_spin)
        
        self.default_grow_spin = QSpinBox()
        self.default_grow_spin.setRange(-10, 10)
        proc_layout.addRow("Default Shrink/Grow:", self.default_grow_spin)
        
        layout.addWidget(proc_group)
        
        # Matting defaults group
        mat_group = QGroupBox("Matting Defaults")
        mat_layout = QFormLayout(mat_group)

        # MatAnyone Internal Resolution selection
        self.default_matany_res_combo = QComboBox()
        self.default_matany_res_combo.addItems(["480", "720", "1080", "1440", "2160", "Full"])
        mat_layout.addRow("MatAnyone Internal Resolution:", self.default_matany_res_combo)
        
        self.default_matany_gamma_spin = QDoubleSpinBox()
        self.default_matany_gamma_spin.setRange(0.1, 10.0)
        self.default_matany_gamma_spin.setSingleStep(0.1)
        self.default_matany_gamma_spin.setDecimals(1)
        mat_layout.addRow("Default Gamma:", self.default_matany_gamma_spin)
        
        self.default_matany_grow_spin = QSpinBox()
        self.default_matany_grow_spin.setRange(-10, 10)
        mat_layout.addRow("Default Shrink/Grow:", self.default_matany_grow_spin)
        
        layout.addWidget(mat_group)

        # Object Removal defaults group
        removal_group = QGroupBox("Object Removal Defaults")
        removal_layout = QFormLayout(removal_group)
        
        self.default_removal_method_combo = QComboBox()
        self.default_removal_method_combo.addItems(["MiniMax-Remover", "OpenCV"])
        removal_layout.addRow("Method:", self.default_removal_method_combo)
        
        self.default_inpaint_grow_spin = QSpinBox()
        self.default_inpaint_grow_spin.setRange(-20, 20)
        removal_layout.addRow("Default Shrink/Grow:", self.default_inpaint_grow_spin)
        
        layout.addWidget(removal_group)
        
        # OpenCV Removal defaults group
        opencv_group = QGroupBox("OpenCV Removal Defaults")
        opencv_layout = QFormLayout(opencv_group)
        
        self.default_opencv_algorithm_combo = QComboBox()
        self.default_opencv_algorithm_combo.addItems(["Telea", "Navier Strokes"])
        opencv_layout.addRow("Algorithm:", self.default_opencv_algorithm_combo)
        
        self.default_opencv_radius_spin = QSpinBox()
        self.default_opencv_radius_spin.setRange(1, 10)
        opencv_layout.addRow("Default Inpaint Radius:", self.default_opencv_radius_spin)
        
        layout.addWidget(opencv_group)
        
        # Minimax-Remover defaults group
        minimax_group = QGroupBox("MiniMax-Remover Defaults")
        minimax_layout = QFormLayout(minimax_group)
        
        self.default_minimax_res_combo = QComboBox()
        self.default_minimax_res_combo.addItems(["352", "480", "720", "1080"])
        minimax_layout.addRow("Internal Resolution:", self.default_minimax_res_combo)
        
        self.default_minimax_vae_tiling_cb = QCheckBox()
        minimax_layout.addRow("Use VAE Tiling:", self.default_minimax_vae_tiling_cb)
        
        self.default_minimax_steps_spin = QSpinBox()
        self.default_minimax_steps_spin.setRange(4, 12)
        minimax_layout.addRow("Default Steps:", self.default_minimax_steps_spin)
        
        layout.addWidget(minimax_group)
        
        return tab
    
    def _create_general_tab(self):
        """Create the general settings tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout(model_group)
        model_layout.addRow("", QLabel("Model changes require restart"))
        
        # SAM model selection
        self.sam_model_combo = QComboBox()
        self.sam_model_combo.addItems(["Base", "Large"])
        self.sam_model_combo.setToolTip("Base model is faster but less accurate.")
        model_layout.addRow("SAM Model:", self.sam_model_combo)
        
        # Force CPU checkbox
        self.force_cpu_cb = QCheckBox()
        self.force_cpu_cb.setToolTip("Force processing to use CPU instead of GPU. Only used for debugging purposes.")
        model_layout.addRow("Force CPU Processing:", self.force_cpu_cb)
        
        layout.addWidget(model_group)
        
        # Frame extraction settings group
        frame_group = QGroupBox("Video frame extraction")
        frame_layout = QFormLayout(frame_group)
        self.frame_format_combo = QComboBox()
        self.frame_format_combo.addItems(["png", "jpg"])
        self.frame_format_combo.setToolTip("PNG is lossless but requires more storage space and is slower than JPG")
        frame_layout.addRow("File Type:", self.frame_format_combo)
        
        layout.addWidget(frame_group)
        
        # Display settings group
        display_group = QGroupBox("Display Settings")
        display_layout = QFormLayout(display_group)
        
        # Display update frequency slider
        self.display_update_slider = QSlider(Qt.Horizontal)
        self.display_update_slider.setRange(1, 10)
        self.display_update_slider.setValue(5)  # Default value
        self.display_update_slider.setToolTip("How often to update the display during tracking or matting.\nLower values are more responsive but slightly slower.")
        self.display_update_slider.setTickPosition(QSlider.TicksBelow)
        self.display_update_slider.setTickInterval(1)
        
        # Create a layout with slider and value label
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.display_update_slider)
        self.display_update_label = QLabel("5")
        self.display_update_label.setFixedWidth(20)
        slider_layout.addWidget(self.display_update_label)
        
        # Update label when slider changes
        self.display_update_slider.valueChanged.connect(
            lambda v: self.display_update_label.setText(str(v))
        )
        
        display_layout.addRow("Display Update Frequency:", slider_layout)
        layout.addWidget(display_group)
        
        # Deduplication threshold settings group
        deduplication_group = QGroupBox("Deduplication Threshold")
        deduplication_layout = QFormLayout(deduplication_group)
        
        self.deduplication_threshold_spin = QDoubleSpinBox()
        self.deduplication_threshold_spin.setRange(0,1)
        self.deduplication_threshold_spin.setValue(0.8)
        self.deduplication_threshold_spin.setSingleStep(0.01)
        self.deduplication_threshold_spin.setToolTip("Value from 0 to 1 used as a threshold for how similar masks have to be before they're considered the same and deduped.\nHigher values are more strict.")
        deduplication_layout.addRow("Threshold:", self.deduplication_threshold_spin)
        
        layout.addWidget(deduplication_group)

        return tab
    
    def _load_current_values(self):
        """Load current settings values into the dialog"""
        app_settings = self.settings_mgr.app_settings
        
        # Defaults tab        
        self.default_show_masks_cb.setChecked(app_settings.default_show_masks)
        self.default_show_outlines_cb.setChecked(app_settings.default_show_outlines)
        self.default_antialias_cb.setChecked(app_settings.default_antialias)
        self.default_color_picker.set_color(app_settings.default_bgcolor)
        
        self.default_holes_spin.setValue(app_settings.default_holes)
        self.default_dots_spin.setValue(app_settings.default_dots)
        self.default_border_spin.setValue(app_settings.default_border_fix)
        self.default_grow_spin.setValue(app_settings.default_grow)
        
        self.default_matany_gamma_spin.setValue(app_settings.default_matany_gamma)
        self.default_matany_grow_spin.setValue(app_settings.default_matany_grow)

        # Set MatAnyone resolution combo box
        if app_settings.default_matany_res == 0:
            self.default_matany_res_combo.setCurrentText("Full")
        else:
            self.default_matany_res_combo.setCurrentText(str(app_settings.default_matany_res))
        
        # Object Removal defaults
        self.default_removal_method_combo.setCurrentText(app_settings.default_removal_method)
        self.default_inpaint_grow_spin.setValue(app_settings.default_inpaint_grow)
        
        # OpenCV Removal defaults
        self.default_opencv_algorithm_combo.setCurrentText(app_settings.default_inpaint_method)
        self.default_opencv_radius_spin.setValue(app_settings.default_inpaint_radius)
        
        # Minimax-Remover defaults
        self.default_minimax_res_combo.setCurrentText(str(app_settings.default_minimax_resolution))
        self.default_minimax_vae_tiling_cb.setChecked(app_settings.default_minimax_vae_tiling)
        self.default_minimax_steps_spin.setValue(app_settings.default_minimax_steps)

        # General tab
        self.sam_model_combo.setCurrentText(app_settings.sam_model)
        self.force_cpu_cb.setChecked(app_settings.force_cpu)
        self.frame_format_combo.setCurrentText(app_settings.frame_format)
        self.display_update_slider.setValue(app_settings.display_update_frequency)
        self.display_update_label.setText(str(app_settings.display_update_frequency))
        self.deduplication_threshold_spin.setValue(app_settings.dedupe_threshold)
    
    
    def _save_current_values(self):
        """Save dialog values back to settings"""
        app_settings = self.settings_mgr.app_settings
        
        # Defaults tab
        app_settings.default_show_masks = self.default_show_masks_cb.isChecked()
        app_settings.default_show_outlines = self.default_show_outlines_cb.isChecked()
        app_settings.default_antialias = self.default_antialias_cb.isChecked()
        app_settings.default_bgcolor = self.default_color_picker.color_rgb
        app_settings.default_holes = self.default_holes_spin.value()
        app_settings.default_dots = self.default_dots_spin.value()
        app_settings.default_border_fix = self.default_border_spin.value()
        app_settings.default_grow = self.default_grow_spin.value()
        app_settings.default_matany_gamma = self.default_matany_gamma_spin.value()
        app_settings.default_matany_grow = self.default_matany_grow_spin.value()

        # Handle MatAnyone resolution setting
        matany_text = self.default_matany_res_combo.currentText()
        if matany_text == "Full":
            app_settings.default_matany_res = 0
        else:
            app_settings.default_matany_res = int(matany_text)
        
        # Object Removal defaults
        app_settings.default_removal_method = self.default_removal_method_combo.currentText()
        app_settings.default_inpaint_grow = self.default_inpaint_grow_spin.value()
        
        # OpenCV Removal defaults
        app_settings.default_inpaint_method = self.default_opencv_algorithm_combo.currentText()
        app_settings.default_inpaint_radius = self.default_opencv_radius_spin.value()
        
        # Minimax-Remover defaults
        app_settings.default_minimax_resolution = int(self.default_minimax_res_combo.currentText())
        app_settings.default_minimax_vae_tiling = self.default_minimax_vae_tiling_cb.isChecked()
        app_settings.default_minimax_steps = self.default_minimax_steps_spin.value()

        # General tab
        app_settings.sam_model = self.sam_model_combo.currentText()
        app_settings.force_cpu = self.force_cpu_cb.isChecked()
        app_settings.frame_format = self.frame_format_combo.currentText()
        app_settings.display_update_frequency = self.display_update_slider.value()
        app_settings.dedupe_threshold = self.deduplication_threshold_spin.value()
            
    
    def accept(self):
        """Apply settings and close dialog"""
        # Check what changed before saving
        old_settings = self._original_app_settings
        self._save_current_values()
        new_settings = self.settings_mgr.app_settings
        
        # Check for changes that require refreshing
        self._handle_setting_changes(old_settings, new_settings)
        
        self.settings_mgr.save_app_settings()
        super().accept()
    
    def _handle_setting_changes(self, old_settings, new_settings):
        """Handle changes that require special actions"""
        # Check for splitter size changes
        splitter_changed = (
            old_settings.main_splitter_sizes != new_settings.main_splitter_sizes or
            old_settings.vertical_splitter_sizes != new_settings.vertical_splitter_sizes or
            old_settings.bottom_splitter_sizes != new_settings.bottom_splitter_sizes
        )
        
        # Notify parent window if it exists
        if hasattr(self, 'parent') and self.parent():
            if splitter_changed:
                # Call method to refresh splitter sizes
                if hasattr(self.parent(), '_refresh_splitter_sizes'):
                    self.parent()._refresh_splitter_sizes()

    
    def reject(self):
        """Cancel changes and close dialog"""
        self._restore_app_settings()
        super().reject()
