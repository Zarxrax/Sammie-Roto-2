import sys
import os
import argparse
import json
import traceback
import webbrowser
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QWidget, QPushButton, QLabel, QStatusBar, QSlider, 
    QTabWidget, QSpinBox, QComboBox, QSplitter, QGroupBox, QTextEdit,
    QCheckBox, QLineEdit, QMessageBox, QProgressDialog, QDialog
)
from PySide6.QtGui import (
    QPixmap, QAction, QShortcut, QKeySequence, QTextCursor, QIcon, QFont
)
from PySide6.QtCore import Qt, QTimer

# Import external logic functions
from sammie import sammie
from sammie import resources
from sammie.export_image_dialog import ImageExportDialog
from sammie.export_dialog import ExportDialog
from sammie.settings_dialog import SettingsDialog
from sammie.settings_manager import get_settings_manager, initialize_settings, ApplicationSettings

# Import GUI widgets
from sammie.gui_widgets import (
    ConsoleRedirect, ColorDisplayWidget, UpdateChecker, ClickableLabel,
    HotkeysHelpDialog, PointTable, ImageViewer, ColorPickerWidget
)

# ==================== VERSION ====================

__version__ = "2.0.0"

# ==================== LOGGING HELPER ====================

def log_exception(exc_type, exc_value, exc_traceback):
    """Simple exception logger"""
    # Format the exception message
    error_message = f"\nERROR at {datetime.now()}:\n"
    error_message += f"{exc_type.__name__}: {exc_value}\n"
    
    # Get the traceback as a string
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    error_message += ''.join(tb_lines)
    error_message += "-" * 50 + "\n"
    
    try:
        with open("sammie_debug.log", "w", encoding="utf-8") as f:
            f.write(error_message)
    except:
        pass  # Don't let logging errors crash the app
        
    # Also send to console redirect if it exists
    if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'write'):
        try:
            sys.stdout.write(error_message)
            sys.stdout.flush()
        except:
            pass


# ==================== TAB WIDGETS ====================

class SegmentationTab(QWidget):
    """Tab containing segmentation controls and parameters"""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the segmentation tab layout"""
        layout = QVBoxLayout(self)
        
        # Add Point group
        self._create_add_point_group(layout)
        
        # Clear Points group
        self._create_clear_points_group(layout)
        
        # Tracking group
        self._create_tracking_group(layout)
        
        # Parameter sliders (renamed to Postprocessing)
        self._create_parameter_sliders(layout)
        
        layout.addStretch()
    
    def _create_add_point_group(self, layout):
        """Create the Add Point group with object selector and point type"""
        add_point_group = QGroupBox("Add Point")
        add_point_layout = QVBoxLayout(add_point_group)
        
        # Object selector with color display
        object_row = QHBoxLayout()
        object_row.addWidget(QLabel("Object:"))
        
        self.object_spinbox = QSpinBox()
        self.object_spinbox.setRange(0, 20)
        self.object_spinbox.setValue(0)
        self.object_spinbox.setToolTip("Select which object ID to assign to new points. Each object gets a unique color.")
        object_row.addWidget(self.object_spinbox)
        
        # Color display widget
        self.color_display = ColorDisplayWidget(sammie.PALETTE[0])
        object_row.addWidget(self.color_display)
        object_row.addStretch()
        
        add_point_layout.addLayout(object_row)
        
        # Object name input
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        
        self.object_name_input = QLineEdit()
        self.object_name_input.setPlaceholderText("Enter object name...")
        self.object_name_input.setMaxLength(50)  # Reasonable limit
        self.object_name_input.setToolTip("Optional: Give this object a name for easier identification")
        name_row.addWidget(self.object_name_input)
        
        add_point_layout.addLayout(name_row)
        
        # Connect signals
        self.object_spinbox.valueChanged.connect(self._update_color_display)
        self.object_spinbox.valueChanged.connect(self._update_name_display)
        self.object_name_input.textChanged.connect(self._save_object_name)
        
        # Load initial name
        self._update_name_display(0)
        
        # Instructions for mouse clicks
        instructions_label = QLabel("Left-click: Add positive point\nRight-click: Add negative point")
        instructions_label.setStyleSheet("""
            QLabel {
                background-color: palette(alternate-base);
                padding: 10px;
                border: 1px solid palette(mid);
                border-radius: 5px;
                font-size: 11px;
                line-height: 1.3;
            }
        """)

        instructions_label.setAlignment(Qt.AlignCenter)
        add_point_layout.addWidget(instructions_label)
        
        layout.addWidget(add_point_group)
    
    def _create_clear_points_group(self, layout):
        """Create the Clear Points group with all clearing actions"""
        clear_group = QGroupBox("Clear Points")
        clear_layout = QVBoxLayout(clear_group)
        
        button_configs = [
            ("Remove Last Point", "undo_last_point_btn", 
            "Remove the most recently added point"),
            ("Clear Frame", "clear_frame_btn", 
            "Remove all points from the current frame"),
            ("Clear Object", "clear_object_btn", 
            "Remove all points for the currently selected object"),
            ("Clear All", "clear_all_btn", 
            "Remove all points from all frames and objects")
        ]
        
        for btn_text, attr_name, tooltip in button_configs:
            btn = QPushButton(btn_text)
            btn.setToolTip(tooltip)
            clear_layout.addWidget(btn)
            setattr(self, attr_name, btn)
        
        layout.addWidget(clear_group)
    
    def _create_tracking_group(self, layout):
        """Create the Tracking group with tracking-related actions"""
        tracking_group = QGroupBox("Tracking")
        tracking_layout = QVBoxLayout(tracking_group)
        
        button_configs = [
            (" Track Objects ", "track_objects_btn", 
            "Propagate segmentation masks to all frames using the added points as guidance"),
            ("Clear Tracking Data", "clear_tracking_data_btn", 
            "Remove all propagated masks but keep the point annotations"),
            (" Deduplicate Similar Masks ", "deduplicate_masks_btn", 
            "Reduce edge chatter in animated content by repeating masks on duplicated frames (requires tracking first)")
        ]
        
        for btn_text, attr_name, tooltip in button_configs:
            btn = QPushButton(btn_text)
            btn.setToolTip(tooltip)
            tracking_layout.addWidget(btn)
            setattr(self, attr_name, btn)
        
        # Put icon on the right side of text
        self.track_objects_btn.setLayoutDirection(Qt.RightToLeft)
        self.deduplicate_masks_btn.setLayoutDirection(Qt.RightToLeft)

        layout.addWidget(tracking_group)
    
    def _create_parameter_sliders(self, layout):
        """Create parameter adjustment sliders"""
        sliders_group = QGroupBox("Postprocessing")
        sliders_layout = QGridLayout(sliders_group)
        
        settings_mgr = get_settings_manager()
        slider_configs = [
            ("Remove Holes:", 0, 50, 0, settings_mgr.get_session_setting("holes", 0), "holes",
            "Remove small holes inside segmented objects."),
            ("Remove Dots:", 0, 50, 0, settings_mgr.get_session_setting("dots", 0), "dots",
            "Remove small isolated regions outside main objects."),
            ("Border Fix:", 0, 10, 0, settings_mgr.get_session_setting("border_fix", 0), "border_fix",
            "Fix artifacts at the edge of the frame by extending masks towards the edge."),
            ("Shrink/Grow:", -10, 10, 0, settings_mgr.get_session_setting("grow", 0), "grow",
            "Shrink (erode) or grow (dilate) the segmented regions.")
        ]
        
        for i, (label_text, min_val, max_val, default_val, current_val, attr_prefix, tooltip) in enumerate(slider_configs):
            # Create clickable label for reset functionality
            label = ClickableLabel(label_text)
            label.setToolTip(f"Double-click to reset to default value ({default_val})")
            sliders_layout.addWidget(label, i, 0)
            
            # Create slider with tooltip
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(current_val)
            slider.setToolTip(tooltip)
            sliders_layout.addWidget(slider, i, 1)
            
            # Create value display
            value_label = QLabel(str(current_val))
            value_label.setMinimumWidth(30)
            value_label.setAlignment(Qt.AlignCenter)
            sliders_layout.addWidget(value_label, i, 2)
            
            # Connect slider to value display and save settings
            slider.valueChanged.connect(
                lambda v, lbl=value_label: lbl.setText(str(v))
            )
            slider.valueChanged.connect(
                lambda v, key=attr_prefix: self._save_slider_value(key, v)
            )
            
            # Connect label double-click to reset slider
            label.doubleClicked.connect(
                lambda default=default_val, s=slider: self._reset_slider_to_default(s, default)
            )
            
            # Store references
            setattr(self, f"{attr_prefix}_slider", slider)
            setattr(self, f"{attr_prefix}_value", value_label)
        
        layout.addWidget(sliders_group)
    
    def _reset_slider_to_default(self, slider, default_value):
        """Reset a slider to its default value"""
        slider.setValue(default_value)

    def _save_slider_value(self, key, value):
        """Save slider value to session settings"""
        settings_mgr = get_settings_manager()
        settings_mgr.set_session_setting(key, value)

    def _update_color_display(self, object_id):
        """Update the color display when object ID changes"""
        if 0 <= object_id < len(sammie.PALETTE):
            self.color_display.set_color(sammie.PALETTE[object_id])

    def get_selected_object_id(self):
        """Get the currently selected object ID"""
        return self.object_spinbox.value()
    
    def _update_name_display(self, object_id):
        """Update the name input field when object ID changes"""
        settings_mgr = get_settings_manager()
        object_names = settings_mgr.get_session_setting("object_names", {})
        current_name = object_names.get(str(object_id), "")
        self.object_name_input.setText(current_name)

    def _save_object_name(self, name):
        """Save object name to session settings"""
        settings_mgr = get_settings_manager()
        object_id = self.object_spinbox.value()
        object_names = settings_mgr.get_session_setting("object_names", {})
        
        if name.strip():
            object_names[str(object_id)] = name.strip()
        elif str(object_id) in object_names:
            # Remove empty names
            del object_names[str(object_id)]
        
        settings_mgr.set_session_setting("object_names", object_names)
        
        # Notify parent window to refresh the point table
        if hasattr(self, 'parent_window') and hasattr(self.parent_window, '_on_object_name_changed'):
            self.parent_window._on_object_name_changed()

    def get_object_name(self, object_id):
        """Get the name for a specific object ID"""
        settings_mgr = get_settings_manager()
        object_names = settings_mgr.get_session_setting("object_names", {})
        return object_names.get(str(object_id), "")
    
        
    def update_tracking_status_helper(self, propagated):
        """Update the Track Objects button icon based on propagation state"""
        if propagated:
            self.track_objects_btn.setIcon(QIcon(":/icons/check-small.png"))
            #self.track_objects_btn.setText("Track Objects ✅")
            self.deduplicate_masks_btn.setEnabled(True)
        else:
            #self.track_objects_btn.setText("Track Objects")
            self.track_objects_btn.setIcon(QIcon())
            self.deduplicate_masks_btn.setEnabled(False)
            self.deduplicate_masks_btn.setIcon(QIcon())
            #self.deduplicate_masks_btn.setText("Deduplicate Similar Masks")

    def update_deduplicate_status_helper(self, deduplicated):
        """Update the Deduplicate button text based on deduplication status"""
        if deduplicated:
            self.deduplicate_masks_btn.setIcon(QIcon(":/icons/check-small.png"))
            #self.deduplicate_masks_btn.setText("Deduplicate Similar Masks ✅")
        else:
            self.deduplicate_masks_btn.setIcon(QIcon())
            #self.deduplicate_masks_btn.setText("Deduplicate Similar Masks")

    def load_values_from_settings(self):
        """Load all values from settings (useful when loading a session)"""
        settings_mgr = get_settings_manager()
        
        # Update object spinbox
        self.object_spinbox.setValue(0)
        
        # Update object name
        self._update_name_display(0)
        
        # Update sliders
        slider_mappings = [
            ("holes", self.holes_slider, self.holes_value),
            ("dots", self.dots_slider, self.dots_value),
            ("border_fix", self.border_fix_slider, self.border_fix_value),
            ("grow", self.grow_slider, self.grow_value)
        ]
        
        for setting_key, slider, value_label in slider_mappings:
            value = settings_mgr.get_session_setting(setting_key, 0)
            slider.setValue(value)
            value_label.setText(str(value))


class MattingTab(QWidget):
    """Tab containing matting controls and parameters"""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the matting tab layout"""
        layout = QVBoxLayout(self)
        
        # Instructions
        self._create_instructions_section(layout)
        
        # Run button
        self.run_matting_btn = QPushButton(" Run Matting ")
        self.run_matting_btn.setLayoutDirection(Qt.RightToLeft)
        layout.addWidget(self.run_matting_btn)
        
        # MatAnyone Internal Resolution selection
        res_group = QGroupBox("Resolution Settings")
        res_layout = QHBoxLayout(res_group)
        
        res_label = QLabel("Internal Resolution:")
        self.matany_res_combo = QComboBox()
        self.matany_res_combo.addItems(["480", "720", "1080", "1440", "2160", "Full"])
        self.matany_res_combo.setToolTip("If your video's resolution is higher than this, it will be\ndownsampled to this resolution before running matting.\nThis reduces VRAM requirements and increases processing speed.")
        
        # Connect to save settings when changed
        self.matany_res_combo.currentTextChanged.connect(self._save_resolution_setting)
        
        res_layout.addWidget(res_label)
        res_layout.addWidget(self.matany_res_combo)
        res_layout.addStretch()
        
        layout.addWidget(res_group)

        # Parameters
        self._create_parameter_sliders(layout)
        
        layout.addStretch()
    
    def _create_instructions_section(self, layout):
        """Create the instructions section for the matting tab"""
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        
        # Create the instruction text
        instructions_text = QLabel()
        instructions_text.setWordWrap(True)
        instructions_text.setTextFormat(Qt.RichText)  # Allow HTML formatting
        
        # Set the instruction content
        instruction_content = """
        • Matting can be used to create mattes for objects with soft or poorly defined edges.<br>
        • You first need to add points to at least one frame in the Segmentation tab, then press the 'Run Matting' button.<br>
        • If you add points to multiple frames, the matting will 'refresh' at each keyframe, which may momentarily break temporal stability.<br>
        • Each object is processed separately, so multiple objects will multiply the processing time.<br>
        • Tracking objects in the Segmentation tab is <b>not</b> needed and will <b>not</b> improve the result.<br>
        """
        
        instructions_text.setText(instruction_content)
        
        # Style the text
        instructions_text.setStyleSheet("""
            QLabel {
                background-color: palette(alternate-base);
                padding: 10px;
                border: 1px solid palette(mid);
                border-radius: 5px;
                font-size: 11px;
                line-height: 1.3;
            }
        """)
        
        instructions_layout.addWidget(instructions_text)
        layout.addWidget(instructions_group)

    def _create_parameter_sliders(self, layout):
        """Create parameter adjustment sliders for matting"""
        sliders_group = QGroupBox("Postprocessing")
        sliders_layout = QGridLayout(sliders_group)
        
        settings_mgr = get_settings_manager()

        # Define slider configurations with tooltips and default values from settings
        slider_configs = [
            ("Gamma:", 1, 1000, "matany_gamma", 1.0,
            "Values < 1.0 darken edges, values > 1.0 brighten edges.",
            lambda v: f"{v/100.0:.1f}", lambda v: int(v * 100), lambda v: v / 100.0),
            ("Shrink/Grow:", -10, 10, "matany_grow", 0,
            "Shrink (erode) or grow (dilate) the matted regions.",
            lambda v: str(v), lambda v: v, lambda v: v)
        ]

        for i, (label_text, min_val, max_val, setting_key, fallback_default, tooltip, 
                display_func, slider_func, save_func) in enumerate(slider_configs):
            
            # Get default from settings manager
            default_val = getattr(settings_mgr.app_settings, f"default_{setting_key}", fallback_default)
            current_val = settings_mgr.get_session_setting(setting_key, default_val)
            
            # Create clickable label for reset functionality
            label = ClickableLabel(label_text)
            label.setToolTip(f"Double-click to reset to default value ({display_func(slider_func(default_val))})")
            sliders_layout.addWidget(label, i, 0)
            
            # Create slider with tooltip
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(slider_func(current_val))
            slider.setToolTip(tooltip)
            sliders_layout.addWidget(slider, i, 1)
            
            # Create value display
            value_label = QLabel(display_func(slider_func(current_val)))
            value_label.setMinimumWidth(35 if setting_key == "matany_gamma" else 30)
            value_label.setAlignment(Qt.AlignCenter)
            sliders_layout.addWidget(value_label, i, 2)
            
            # Connect slider to value display and save settings
            if setting_key == "matany_gamma":
                slider.valueChanged.connect(self._update_gamma_value)
                slider.valueChanged.connect(
                    lambda v, func=save_func: self._save_slider_value("matany_gamma", func(v))
                )
                # Connect label double-click to reset slider
                label.doubleClicked.connect(
                    lambda default=default_val: self._reset_gamma_to_default(default)
                )
                # Store references
                self.gamma_slider = slider
                self.gamma_value = value_label
            else:
                slider.valueChanged.connect(
                    lambda v, lbl=value_label, func=display_func: lbl.setText(func(v))
                )
                slider.valueChanged.connect(
                    lambda v, key=setting_key, func=save_func: self._save_slider_value(key, func(v))
                )
                # Connect label double-click to reset slider
                label.doubleClicked.connect(
                    lambda s=slider, default=default_val, func=slider_func: self._reset_slider_to_default(s, func(default))
                )
                # Store references
                self.shrink_grow_slider = slider
                self.shrink_grow_value = value_label
        
        layout.addWidget(sliders_group)

    def _reset_gamma_to_default(self, default_value):
        """Reset gamma slider to its default value"""
        self.gamma_slider.setValue(int(default_value * 100))

    def _reset_slider_to_default(self, slider, default_value):
        """Reset a slider to its default value"""
        slider.setValue(default_value)
        
    def _update_gamma_value(self, value):
        """Update gamma value display (convert from int to decimal)"""
        gamma_val = value / 100.0
        self.gamma_value.setText(f"{gamma_val:.1f}")
    
    def _save_resolution_setting(self, value):
        """Save resolution combo box value to session settings"""
        if value == "Full":
            resolution = 0
        else:
            resolution = int(value)
        settings_mgr = get_settings_manager()
        settings_mgr.set_session_setting("matany_res", resolution)
        
    def _save_slider_value(self, key, value):
        """Save slider value to session settings"""
        settings_mgr = get_settings_manager()
        settings_mgr.set_session_setting(key, value)

    def load_values_from_settings(self):
        """Load all values from settings"""
        settings_mgr = get_settings_manager()
        
        # Load resolution
        resolution = settings_mgr.get_session_setting("matany_res", 720)

        if resolution == 0:
            self.matany_res_combo.setCurrentText("Full")
        else:
            index = self.matany_res_combo.findText(str(resolution))
            if index >= 0:
                self.matany_res_combo.setCurrentIndex(index)

        # Update gamma slider
        gamma = settings_mgr.get_session_setting("matany_gamma", 1.0)
        self.gamma_slider.setValue(int(gamma * 100))
        self.gamma_value.setText(f"{gamma:.1f}")
        
        # Update shrink/grow slider
        shrink_grow = settings_mgr.get_session_setting("matany_grow", 0)
        self.shrink_grow_slider.setValue(shrink_grow)
        self.shrink_grow_value.setText(str(shrink_grow))

    def update_matting_status(self, is_propagated):
        """Update the Run Matting button text based on propagation state"""
        if is_propagated:
            self.run_matting_btn.setIcon(QIcon(":/icons/check-small.png"))
            #self.run_matting_btn.setText("Run Matting ✅")
        else:
            self.run_matting_btn.setIcon(QIcon())
            #self.run_matting_btn.setText("Run Matting")

class ObjectRemovalTab(QWidget):
    """Tab containing object removal controls and parameters"""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the object removal tab layout"""
        layout = QVBoxLayout(self)
        
        # Instructions
        self._create_instructions_section(layout)
        
        # Run button
        self.run_removal_btn = QPushButton(" Run Object Removal ")
        self.run_removal_btn.setLayoutDirection(Qt.RightToLeft)
        layout.addWidget(self.run_removal_btn)

        # Method selection (MiniMax-Remover vs OpenCV)
        self._create_method_selection(layout)

        # Create both parameter groups (they'll be shown/hidden based on method)
        self._create_opencv_parameters(layout)
        self._create_minimax_parameters(layout)
        
        # Create shared shrink/grow slider first (used by both methods)
        self._create_shared_shrink_grow(layout)

        # Show appropriate parameters for initial method
        self._update_parameters_visibility()

        layout.addStretch()
    
    def _create_instructions_section(self, layout):
        """Create the instructions section for the object removal tab"""
        instructions_group = QGroupBox("Instructions")
        instructions_layout = QVBoxLayout(instructions_group)
        
        # Create the instruction text
        instructions_text = QLabel()
        instructions_text.setWordWrap(True)
        instructions_text.setTextFormat(Qt.RichText)
        
        # Set the instruction content
        instruction_content = """
        • Object removal uses inpainting to fill in areas where objects have been removed.<br>
        • You first need to run tracking in the Segmentation tab, so a mask is on every frame.<br>
        • The OpenCV option is really bad, and is only provided as a fallback in case MiniMax-Remover can't be used.<br>
        """
        
        instructions_text.setText(instruction_content)
        
        # Style the text
        instructions_text.setStyleSheet("""
            QLabel {
                background-color: palette(alternate-base);
                padding: 10px;
                border: 1px solid palette(mid);
                border-radius: 5px;
                font-size: 11px;
                line-height: 1.3;
            }
        """)
        
        instructions_layout.addWidget(instructions_text)
        layout.addWidget(instructions_group)
    
    def _create_method_selection(self, layout):
        """Create method selection (MiniMax-Remover vs OpenCV)"""
        method_group = QGroupBox("Method")
        method_layout = QHBoxLayout(method_group)
        
        method_layout.addWidget(QLabel("Method:"))
        
        self.method_combo = QComboBox()
        self.method_combo.addItems(["MiniMax-Remover", "OpenCV"])
        
        settings_mgr = get_settings_manager()
        current_method = settings_mgr.get_session_setting("removal_method", "MiniMax-Remover")
        index = self.method_combo.findText(current_method)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
        
        self.method_combo.setToolTip("MiniMax-Remover: Uses a video diffusion model (recommended).<br>OpenCV: Uses traditional computing algorithms (poor quality).")
        self.method_combo.currentTextChanged.connect(self._on_method_changed)
        
        method_layout.addWidget(self.method_combo)
        method_layout.addStretch()
        
        layout.addWidget(method_group)
    
    def _on_method_changed(self, method):
        """Handle method selection change"""
        settings_mgr = get_settings_manager()
        settings_mgr.set_session_setting("removal_method", method)
        self._update_parameters_visibility()
    
    def _update_parameters_visibility(self):
        """Show/hide parameter groups based on selected method"""
        current_method = self.method_combo.currentText()
        
        if current_method == "OpenCV":
            self.opencv_params_group.setVisible(True)
            self.minimax_params_group.setVisible(False)
        else:  # MiniMax-Remover
            self.opencv_params_group.setVisible(False)
            self.minimax_params_group.setVisible(True)

    def _create_shared_shrink_grow(self, layout):
        """Create the shared shrink/grow slider used by both methods"""
        settings_mgr = get_settings_manager()
        
        shrink_grow_group = QGroupBox("Mask Adjustment")
        shrink_grow_layout = QGridLayout(shrink_grow_group)
        
        default_grow = getattr(settings_mgr.app_settings, "default_inpaint_grow", 5)
        current_grow = settings_mgr.get_session_setting("inpaint_grow", default_grow)
        
        label = ClickableLabel("Shrink/Grow:")
        label.setToolTip(f"Double-click to reset to default value ({default_grow})")
        shrink_grow_layout.addWidget(label, 0, 0)
        
        self.shrink_grow_slider = QSlider(Qt.Horizontal)
        self.shrink_grow_slider.setRange(-20, 20)
        self.shrink_grow_slider.setValue(current_grow)
        self.shrink_grow_slider.setToolTip("Shrink (erode) or grow (dilate) the mask before inpainting. This is additive to the same setting on the Segmentation tab.")
        shrink_grow_layout.addWidget(self.shrink_grow_slider, 0, 1)
        
        self.shrink_grow_value = QLabel(str(current_grow))
        self.shrink_grow_value.setMinimumWidth(30)
        self.shrink_grow_value.setAlignment(Qt.AlignCenter)
        shrink_grow_layout.addWidget(self.shrink_grow_value, 0, 2)
        
        self.shrink_grow_slider.valueChanged.connect(
            lambda v: self.shrink_grow_value.setText(str(v))
        )
        self.shrink_grow_slider.valueChanged.connect(
            lambda v: self._save_slider_value("inpaint_grow", v)
        )
        
        label.doubleClicked.connect(
            lambda: self._reset_slider_to_default(self.shrink_grow_slider, default_grow)
        )
        
        layout.addWidget(shrink_grow_group)
        """Show/hide parameter groups based on selected method"""
        current_method = self.method_combo.currentText()
        
        if current_method == "OpenCV":
            self.opencv_params_group.setVisible(True)
            self.minimax_params_group.setVisible(False)
        else:  # MiniMax-Remover
            self.opencv_params_group.setVisible(False)
            self.minimax_params_group.setVisible(True)

    def _create_opencv_parameters(self, layout):
        """Create parameters for OpenCV method"""
        settings_mgr = get_settings_manager()
        
        self.opencv_params_group = QWidget()
        opencv_layout = QVBoxLayout(self.opencv_params_group)
        opencv_layout.setContentsMargins(0, 0, 0, 0)

        # Algorithm selection
        algorithm_group = QGroupBox("Algorithm")
        algorithm_layout = QHBoxLayout(algorithm_group)
        
        algorithm_layout.addWidget(QLabel("Algorithm:"))
        
        self.opencv_algorithm_combo = QComboBox()
        self.opencv_algorithm_combo.addItems(["Telea", "Navier-Stokes"])
        
        current_algorithm = settings_mgr.get_session_setting("inpaint_algorithm", "Telea")
        index = self.opencv_algorithm_combo.findText(current_algorithm)
        if index >= 0:
            self.opencv_algorithm_combo.setCurrentIndex(index)
        
        self.opencv_algorithm_combo.setToolTip("Telea: Based on fast marching method.\nNavier-Stokes: Fluid dynamics based method, may produce smoother results.")
        self.opencv_algorithm_combo.currentTextChanged.connect(self._save_opencv_algorithm)
        
        algorithm_layout.addWidget(self.opencv_algorithm_combo)
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
                lambda v, key=setting_key, func=save_func: self._save_slider_value(key, func(v))
            )
            
            label.doubleClicked.connect(
                lambda s=slider, default=default_val, func=slider_func: self._reset_slider_to_default(s, func(default))
            )
            
            if setting_key == "inpaint_radius":
                self.opencv_radius_slider = slider
                self.opencv_radius_value = value_label
        
        opencv_layout.addWidget(sliders_group)
        layout.addWidget(self.opencv_params_group)

    def _create_minimax_parameters(self, layout):
        """Create parameters for MiniMax-Remover method"""
        settings_mgr = get_settings_manager()

        self.minimax_params_group = QWidget()
        minimax_layout = QVBoxLayout(self.minimax_params_group)
        minimax_layout.setContentsMargins(0, 0, 0, 0)
        
        params_group = QGroupBox("Parameters")
        params_layout = QGridLayout(params_group)
        
        row = 0
        
        # Internal Resolution
        params_layout.addWidget(QLabel("Internal Resolution:"), row, 0)
        self.minimax_resolution_combo = QComboBox()
        self.minimax_resolution_combo.addItems(["352", "480", "720", "1080"])
        
        current_resolution = str(settings_mgr.get_session_setting("minimax_resolution", 480))
        index = self.minimax_resolution_combo.findText(current_resolution)
        if index >= 0:
            self.minimax_resolution_combo.setCurrentIndex(index)
        
        self.minimax_resolution_combo.setToolTip("Internal processing resolution. Higher values produce better quality but are slower and use more VRAM.")
        self.minimax_resolution_combo.currentTextChanged.connect(
            lambda v: settings_mgr.set_session_setting("minimax_resolution", int(v))
        )
        params_layout.addWidget(self.minimax_resolution_combo, row, 1, 1, 2)
        
        row += 1
        
        # VAE Tiling checkbox
        params_layout.addWidget(QLabel("Use VAE Tiling:"), row, 0)
        self.minimax_vae_tiling_checkbox = QCheckBox()
        
        vae_tiling = settings_mgr.get_session_setting("minimax_vae_tiling", False)
        self.minimax_vae_tiling_checkbox.setChecked(vae_tiling)
        self.minimax_vae_tiling_checkbox.setToolTip("If you get an out of memory error during the VAE decode step, try enabling this option. The VAE steps will take longer but use less VRAM.")
        self.minimax_vae_tiling_checkbox.stateChanged.connect(
            lambda state: settings_mgr.set_session_setting("minimax_vae_tiling", self.minimax_vae_tiling_checkbox.isChecked())
        )
        params_layout.addWidget(self.minimax_vae_tiling_checkbox, row, 1, 1, 2)
        
        row += 1
        
        # Steps slider
        default_steps = getattr(settings_mgr.app_settings, "default_minimax_steps", 6)
        current_steps = settings_mgr.get_session_setting("minimax_steps", default_steps)
        
        label = ClickableLabel("Steps:")
        label.setToolTip(f"Double-click to reset to default value ({default_steps})")
        params_layout.addWidget(label, row, 0)
        
        self.minimax_steps_slider = QSlider(Qt.Horizontal)
        self.minimax_steps_slider.setRange(4, 12)
        self.minimax_steps_slider.setValue(current_steps)
        self.minimax_steps_slider.setToolTip("Number of diffusion steps. Larger values are better quality but slower.")
        params_layout.addWidget(self.minimax_steps_slider, row, 1)
        
        self.minimax_steps_value = QLabel(str(current_steps))
        self.minimax_steps_value.setMinimumWidth(30)
        self.minimax_steps_value.setAlignment(Qt.AlignCenter)
        params_layout.addWidget(self.minimax_steps_value, row, 2)
        
        self.minimax_steps_slider.valueChanged.connect(
            lambda v: self.minimax_steps_value.setText(str(v))
        )
        self.minimax_steps_slider.valueChanged.connect(
            lambda v: settings_mgr.set_session_setting("minimax_steps", v)
        )
        
        label.doubleClicked.connect(
            lambda: self._reset_slider_to_default(self.minimax_steps_slider, default_steps)
        )
        
        minimax_layout.addWidget(params_group)
        layout.addWidget(self.minimax_params_group)

    def _save_opencv_algorithm(self, algorithm):
        """Save OpenCV algorithm to session settings"""
        settings_mgr = get_settings_manager()
        settings_mgr.set_session_setting("inpaint_algorithm", algorithm)

    def _reset_slider_to_default(self, slider, default_value):
        """Reset a slider to its default value"""
        slider.setValue(default_value)
    
    def _save_slider_value(self, key, value):
        """Save slider value to session settings"""
        settings_mgr = get_settings_manager()
        settings_mgr.set_session_setting(key, value)
    
    def load_values_from_settings(self):
        """Load all values from settings"""
        settings_mgr = get_settings_manager()
        
        # Load method selection
        method = settings_mgr.get_session_setting("removal_method", "MiniMax-Remover")
        index = self.method_combo.findText(method)
        if index >= 0:
            self.method_combo.setCurrentIndex(index)
        
        # Update visibility
        self._update_parameters_visibility()
    
        # Load OpenCV settings
        algorithm = settings_mgr.get_session_setting("inpaint_algorithm", "Telea")
        index = self.opencv_algorithm_combo.findText(algorithm)
        if index >= 0:
            self.opencv_algorithm_combo.setCurrentIndex(index)
        
        radius = settings_mgr.get_session_setting("inpaint_radius", 3)
        self.opencv_radius_slider.setValue(radius)
        self.opencv_radius_value.setText(str(radius))
        
        # Load MiniMax settings
        resolution = str(settings_mgr.get_session_setting("minimax_resolution", 480))
        index = self.minimax_resolution_combo.findText(resolution)
        if index >= 0:
            self.minimax_resolution_combo.setCurrentIndex(index)
        
        vae_tiling = settings_mgr.get_session_setting("minimax_vae_tiling", False)
        self.minimax_vae_tiling_checkbox.setChecked(vae_tiling)
        
        steps = settings_mgr.get_session_setting("minimax_steps", 6)
        self.minimax_steps_slider.setValue(steps)
        self.minimax_steps_value.setText(str(steps))
        
        # Load shared shrink/grow setting
        shrink_grow = settings_mgr.get_session_setting("inpaint_grow", 5)
        self.shrink_grow_slider.setValue(shrink_grow)
        self.shrink_grow_value.setText(str(shrink_grow))

    def update_removal_status(self, is_completed):
        """Update the Run Object Removal button text based on completion state"""
        if is_completed:
            self.run_removal_btn.setIcon(QIcon(":/icons/check-small.png"))
            #self.run_removal_btn.setText("Run Object Removal ✅")
        else:
            self.run_removal_btn.setIcon(QIcon())
            #self.run_removal_btn.setText("Run Object Removal")

class Sidebar(QWidget):
    """Main sidebar containing segmentation and matting tabs"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(250)
        self._init_ui()
    
    def _init_ui(self):
        """Initialize the sidebar layout"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        self.tab_widget = QTabWidget()
        self.segmentation_tab = SegmentationTab()
        self.matting_tab = MattingTab()
        self.removal_tab = ObjectRemovalTab()
        
        self.tab_widget.addTab(self.segmentation_tab, "Segmentation")
        self.tab_widget.addTab(self.matting_tab, "Matting")
        self.tab_widget.addTab(self.removal_tab, "Object Removal")
        
        layout.addWidget(self.tab_widget)

    def load_values_from_settings(self):
        """Load values for all tabs from settings"""
        self.segmentation_tab.load_values_from_settings()
        self.matting_tab.load_values_from_settings()
        self.removal_tab.load_values_from_settings()


# ==================== MAIN WINDOW ====================

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, initial_file=None):
        super().__init__()
        self.settings_mgr = initialize_settings()
        self.setWindowTitle(f"Sammie-Roto {__version__}")
        self.setWindowIcon(QIcon(":/icon.ico"))
        self.is_playing = False
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self.play_next_frame)
        self.sam_manager = sammie.SamManager()
        self.matany_manager = sammie.MatAnyManager()
        self.removal_manager = sammie.RemovalManager()
        self.point_manager = sammie.PointManager()
        self.highlighted_point = None
        
        # Store initial file to load after initialization
        self.initial_file = initial_file

        # Connect callbacks for both managers
        self.point_manager.add_callback(self._on_points_changed)
        self.sam_manager.add_callback(self._on_segmentation_changed)
        
        # Initialize update checker
        self.update_checker = UpdateChecker()
        self.update_checker.update_available.connect(self.on_update_available)
        self.update_menu_action = None  # Will store the update menu action when created
        
        # Load session settings
        self.settings_mgr.load_session_settings()

        # Initialize UI
        self._init_ui()
        self._connect_signals()
        self._setup_hotkeys()
        self._update_point_editing_state()
        print(f"Sammie-Roto version {__version__}")
        self.update_checker.check_for_updates()
        sammie.DeviceManager.setup_device()
        self.sam_manager.load_segmentation_model()
        #self.matany_manager.load_matting_model(load_to_cpu=True)
        
        # Load file or resume session
        if self.initial_file:
            if os.path.exists(self.initial_file):
                print(f"Loading file from command line: {self.initial_file}")
                self.load_file(self.initial_file)
            else:
                print(f"Error: Command line file not found: {self.initial_file}")
                self.resume_prev_session()
        else:
            self.resume_prev_session()
    
    # ==================== INITIALIZATION ====================
    
    def _init_ui(self):
        """Initialize the main window UI"""
        self._create_menu_bar()
        self._create_main_layout()
        self._setup_status_bar()
        self._setup_console_redirect()
        
        # Load window settings
        self._load_window_settings()
        
    def _load_window_settings(self):
        """Load window size and position from settings"""
        settings_mgr = self.settings_mgr
        
        width = settings_mgr.app_settings.window_width
        height = settings_mgr.app_settings.window_height
        maximized = settings_mgr.app_settings.window_maximized
        
        self.resize(width, height)
        
        if maximized:
            self.showMaximized()

    def _save_current_ui_state(self):
        """Save current UI state to session settings"""
        settings_mgr = self.settings_mgr
        
        # Save view mode
        settings_mgr.set_session_setting("current_view_mode", self.view_combo.currentText())
        
        # Save checkbox states
        if hasattr(self, 'show_masks_checkbox') and self.show_masks_checkbox:
            settings_mgr.set_session_setting("show_masks", self.show_masks_checkbox.isChecked())
        if hasattr(self, 'show_outlines_checkbox') and self.show_outlines_checkbox:
            settings_mgr.set_session_setting("show_outlines", self.show_outlines_checkbox.isChecked())
        if hasattr(self, 'antialias_checkbox') and self.antialias_checkbox:
            settings_mgr.set_session_setting("antialias", self.antialias_checkbox.isChecked())
        if hasattr(self, 'show_removal_mask_checkbox') and self.show_removal_mask_checkbox:
            settings_mgr.set_session_setting("show_removal_mask", self.show_removal_mask_checkbox.isChecked())
        
        # Save tracking state
        settings_mgr.set_session_setting("is_propagated", self.sam_manager.propagated)
        settings_mgr.set_session_setting("is_deduplicated", self.sam_manager.deduplicated)
        settings_mgr.set_session_setting("is_matted", self.matany_manager.propagated)
        settings_mgr.set_session_setting("is_removed", self.removal_manager.propagated)

    # ==================== SIGNAL CONNECTIONS ====================
        
    def _connect_signals(self):
        """Connect all UI signals"""
        # Connect image viewer point clicks to point addition
        self.viewer.point_clicked.connect(self.add_point_from_click)

        # Connect image viewer file drops to file loading
        self.viewer.file_dropped.connect(self.handle_dropped_file)
        
        # Connect tab widget signals
        self.sidebar.tab_widget.currentChanged.connect(self.on_tab_changed)

        # Get the segmentation tab
        seg_tab = self.sidebar.segmentation_tab
        if seg_tab:
            seg_tab.parent_window = self
            # Connect segmentation tab buttons
            seg_tab.undo_last_point_btn.clicked.connect(self.undo_last_point)
            seg_tab.clear_frame_btn.clicked.connect(self.clear_frame_points)
            seg_tab.clear_object_btn.clicked.connect(self.clear_object_points)
            seg_tab.clear_tracking_data_btn.clicked.connect(self.clear_tracking_data)
            seg_tab.clear_all_btn.clicked.connect(self.clear_all_points)
            seg_tab.track_objects_btn.clicked.connect(self.track_objects)
            seg_tab.deduplicate_masks_btn.clicked.connect(self.deduplicate_similar_masks)

            # Connect postprocessing tab sliders
            seg_tab.holes_slider.valueChanged.connect(self._update_current_frame_display)
            seg_tab.dots_slider.valueChanged.connect(self._update_current_frame_display)
            seg_tab.border_fix_slider.valueChanged.connect(self._update_current_frame_display)
            seg_tab.grow_slider.valueChanged.connect(self._update_current_frame_display)
            
            # Store reference to segmentation tab for status updates
            self.segmentation_tab = seg_tab
            # Initialize the button status
            self.segmentation_tab.update_tracking_status_helper(self.sam_manager.propagated)

        # Get the matting tab
        matting_tab = self.sidebar.matting_tab
        if matting_tab:
            matting_tab.parent_window = self
            # Connect matting tab buttons
            matting_tab.run_matting_btn.clicked.connect(self.run_matting)

            # Connect postprocessing tab sliders
            matting_tab.shrink_grow_slider.valueChanged.connect(self._update_current_frame_display)
            matting_tab.gamma_slider.valueChanged.connect(self._update_current_frame_display)
            
            # Store reference to matting tab for status updates
            self.matting_tab = matting_tab
            # Initialize the button status
            self.matting_tab.update_matting_status(self.matany_manager.propagated)

        # Get the object removal tab
        removal_tab = self.sidebar.removal_tab
        if removal_tab:
            removal_tab.parent_window = self
            # Connect removal tab buttons
            removal_tab.run_removal_btn.clicked.connect(self.run_object_removal)
            
            # Connect method selection to update display
            removal_tab.method_combo.currentTextChanged.connect(self._update_current_frame_display)
            
            # Connect shared shrink/grow slider
            removal_tab.shrink_grow_slider.valueChanged.connect(self._update_current_frame_display)
            
            # Connect OpenCV parameter controls
            removal_tab.opencv_algorithm_combo.currentTextChanged.connect(self._update_current_frame_display)
            removal_tab.opencv_radius_slider.valueChanged.connect(self._update_current_frame_display)
            
            # Connect MiniMax parameter controls
            removal_tab.minimax_resolution_combo.currentTextChanged.connect(self._update_current_frame_display)
            removal_tab.minimax_vae_tiling_checkbox.stateChanged.connect(self._update_current_frame_display)
            removal_tab.minimax_steps_slider.valueChanged.connect(self._update_current_frame_display)
            
            # Store reference to removal tab for status updates
            self.removal_tab = removal_tab
            # Initialize the button status
            self.removal_tab.update_removal_status(self.removal_manager.completed)

    # ==================== UI CREATION ====================

    def _create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        self.file_menu = menubar.addMenu("File")
        file_actions = [
            ("Load Video", self.open_file),
            ("Load Points", self.load_points),
            ("Load Project", self.load_project),
            (None, None),  # Separator
            ("Save Points", self.save_points),
            ("Save Project", self.save_project),
            (None, None),  # Separator
            ("Export Image", self.export_image),
            ("Export Video", self.export_video),
            (None, None),  # Separator
            ("Settings", self.show_settings),
            (None, None),  # Separator
            ("Exit", self.close)
        ]
        
        self._add_menu_actions(self.file_menu, file_actions)
        
        # View menu
        view_menu = menubar.addMenu("View")
        view_actions = [
            ("Fit to Screen", self.fit_to_screen),
            ("100% Zoom", self.zoom_100),
            ("200% Zoom", self.zoom_200),
            ("400% Zoom", self.zoom_400),
            (None, None),  # Separator
            ("Reset Interface", self.reset_interface)
        ]
        
        self._add_menu_actions(view_menu, view_actions)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        help_actions = [
            ("Help", self.show_help),
            ("Shortcut Keys", self.show_hotkeys_help),
            ("About", self.show_about)
        ]
        
        self._add_menu_actions(help_menu, help_actions)
    
    def _add_menu_actions(self, menu, actions):
        """Helper to add actions to a menu"""
        for name, handler in actions:
            if name is None:
                menu.addSeparator()
            else:
                action = QAction(name, self)

                shortcut_map = {
                    self.open_file: "Ctrl+O",
                    self.load_points: "Ctrl+L",
                    self.load_project: "Ctrl+Shift+L",
                    self.save_points: "Ctrl+S",
                    self.save_project: "Ctrl+Shift+S",
                    self.export_video: "Ctrl+E",
                    self.export_image: "Ctrl+Shift+E",
                    self.fit_to_screen: "Ctrl+Backspace",
                    self.zoom_100: "Backspace",
                    self.reset_interface: "Ctrl+Shift+R",
                    self.show_help: "F1",
                    self.show_hotkeys_help: "Ctrl+F1",
                }
                if handler in shortcut_map:
                    action.setShortcut(shortcut_map[handler])

                action.triggered.connect(handler)
                menu.addAction(action)
    
    def _create_main_layout(self):
        """Create the main window layout with splitters"""
        # Main horizontal splitter between viewer and sidebar
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # vertical splitter between viewer and bottom panels
        self.vertical_splitter = QSplitter(Qt.Vertical)
        
        # Center panel with image viewer and controls
        center_panel = self._create_center_panel()
        
        # Bottom panel
        bottom_panel = self._create_bottom_panel()
        
        # Setup vertical splitter
        self.vertical_splitter.addWidget(center_panel)
        self.vertical_splitter.addWidget(bottom_panel)
        self.vertical_splitter.setSizes(self.settings_mgr.app_settings.vertical_splitter_sizes)
        self.vertical_splitter.setCollapsible(0, False)
        self.vertical_splitter.setCollapsible(1, True)
        self.vertical_splitter.setStretchFactor(0, 1)  # viewer
        self.vertical_splitter.setStretchFactor(1, 0)  # bottom panel
        
        # Setup horizontal splitter
        self.sidebar = Sidebar()
        self.sidebar.segmentation_tab.parent_window = self
        self.main_splitter.addWidget(self.vertical_splitter)
        self.main_splitter.addWidget(self.sidebar)
        self.main_splitter.setSizes(self.settings_mgr.app_settings.main_splitter_sizes)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, True)
        self.main_splitter.setStretchFactor(0, 1)  # main area
        self.main_splitter.setStretchFactor(1, 0)  # sidebar
        
        self.setCentralWidget(self.main_splitter)
    
    def _create_center_panel(self):
        """Create the center panel with image viewer and controls"""
        center_panel = QWidget()
        layout = QVBoxLayout(center_panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Image viewer
        self.viewer = ImageViewer(status_callback=self.update_status_bar, parent_window=self)
        layout.addWidget(self.viewer)
        
        # Frame controls
        self._create_frame_controls(layout)
        
        # Playback controls
        self._create_playback_controls(layout)
        
        return center_panel

    def _toggle_show_all_points(self, checked):
        """Toggle show_all_points setting and refresh table"""
        self.settings_mgr.set_session_setting("show_all_points", checked)
        self.settings_mgr.save_session_settings()
        self._update_show_all_points_button_text()
        self._refresh_table()
    
    def _update_show_all_points_button_text(self):
        """Update the button text based on current state"""
        if hasattr(self, 'show_all_points_btn'):
            if self.show_all_points_btn.isChecked():
                self.show_all_points_btn.setText("List All Frame Points")
                self.settings_mgr.set_session_setting("show_all_points", True)
            else:
                self.show_all_points_btn.setText("List Current Frame Points")
                self.settings_mgr.set_session_setting("show_all_points", False)

    def _reset_show_all_points_button_state(self):
        """Update the show_all_points button state from settings"""
        if hasattr(self, 'show_all_points_btn'):
            show_all_points = self.settings_mgr.get_app_setting("default_show_all_points", True)
            self.show_all_points_btn.setChecked(show_all_points)
            self._update_show_all_points_button_text()


    def _create_bottom_panel(self):
        """Create the bottom panel with side-by-side layout"""
        # Create a horizontal splitter for side-by-side layout
        self.bottom_splitter = QSplitter(Qt.Horizontal)
        self.bottom_splitter.setMinimumHeight(100)
        
        # Create containers for each panel
        point_container = QWidget()
        point_layout = QVBoxLayout(point_container)
        point_layout.setContentsMargins(2, 2, 2, 2)
        
        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
        console_layout.setContentsMargins(2, 2, 2, 2)
        
        # Add labels to identify each panel
        point_header_layout = QHBoxLayout()
        point_label = QLabel("Segmentation Point List")
        point_label.setStyleSheet("font-weight: bold; padding: 3px;")
        point_header_layout.addWidget(point_label)
        
        console_label = QLabel("Console")
        console_label.setStyleSheet("font-weight: bold; padding: 3px;")
        console_layout.addWidget(console_label)

        # Add show_all_points toggle button
        self.show_all_points_btn = QPushButton("Show All Frames")
        self.show_all_points_btn.setCheckable(True)
        # Specific styling to overwrite the default blue check toggle
        self.show_all_points_btn.setStyleSheet("""
            QPushButton:checked {
                background-color: palette(button);
            }
        """)
        self.show_all_points_btn.setToolTip("Toggle to show all points for all frames or just the points for the currently displayed frame")
        self.show_all_points_btn.setFixedWidth(150)
        
        # Load initial state from settings
        show_all_points = self.settings_mgr.get_session_setting("show_all_points", True)
        self.show_all_points_btn.setChecked(show_all_points)
        self._update_show_all_points_button_text()
        self.show_all_points_btn.clicked.connect(self._toggle_show_all_points)
        point_header_layout.addStretch()  # Push button to the right
        point_header_layout.addWidget(self.show_all_points_btn)
        point_layout.addLayout(point_header_layout)
        
        # Point table
        self.point_table = PointTable()
        self.point_table.parent_window = self
        # Initialize current_frame in point table
        self.point_table.set_current_frame(self.frame_slider.value())
        point_layout.addWidget(self.point_table)
        self.point_table.point_selected.connect(self._on_point_selected)
        
        # Console
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        console_layout.addWidget(self.console)
        
        console_font = QFont("Consolas")  # Try Consolas first
        console_font.setStyleHint(QFont.Monospace)  # Fallback to system monospace
        self.console.setFont(console_font)
        
        # Add containers to splitter
        self.bottom_splitter.addWidget(point_container)
        self.bottom_splitter.addWidget(console_container)
        
        self.bottom_splitter.setSizes(self.settings_mgr.app_settings.bottom_splitter_sizes)
        
        # Allow both panels to be resized but not collapsed
        self.bottom_splitter.setCollapsible(0, True)
        self.bottom_splitter.setCollapsible(1, True)
        
        return self.bottom_splitter
    
    def _create_frame_controls(self, layout):
        """Create frame navigation controls"""
        slider_layout = QHBoxLayout()
        
        slider_layout.addWidget(QLabel("Frame:"))
        
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        slider_layout.addWidget(self.frame_slider)
        
        self.frame_value = QLabel("0")
        self.frame_value.setMinimumWidth(50)
        self.frame_value.setAlignment(Qt.AlignCenter)
        slider_layout.addWidget(self.frame_value)
        
        self.frame_slider.valueChanged.connect(self.on_frame_change)
        layout.addLayout(slider_layout)
    
    def _create_playback_controls(self, layout):
        """Create playback control buttons"""
        controls_layout = QHBoxLayout()
        
        # Button Icons
        self.icon_play = QIcon(":/icons/control-play.png")
        self.icon_pause = QIcon(":/icons/control-pause.png")
        icon_prev = QIcon(":/icons/control-step-left.png")
        icon_next = QIcon(":/icons/control-step-right.png")
        icon_prev_keyframe = QIcon(":/icons/key-arrow-left.png")
        icon_next_keyframe = QIcon(":/icons/key-arrow-right.png")

        # Playback buttons
        button_configs = [
            (icon_prev_keyframe, 40, self.prev_keyframe, "prev_keyframe", "Previous Keyframe"),
            (icon_prev, 40, self.prev_frame, "prev_frame", "Previous Frame"),
            (self.icon_play, 40, self.toggle_play_pause, "play_pause", "Play/Pause"),
            (icon_next, 40, self.next_frame, "next_frame", "Next Frame"),
            (icon_next_keyframe, 40, self.next_keyframe, "next_keyframe", "Next Keyframe")
        ]
        
        for icon, width, handler, button_id, tooltip in button_configs:
            btn = QPushButton()
            btn.setIcon(icon)
            btn.setMaximumWidth(width)
            btn.setToolTip(tooltip)
            btn.clicked.connect(handler)
            controls_layout.addWidget(btn)
            
            if button_id == "play_pause":
                self.play_pause_btn = btn
        
        controls_layout.addStretch()
        
        # Container for view controls (checkboxes + combobox)
        view_controls_layout = QHBoxLayout()
        
        # Dynamic widgets container
        self.dynamic_widgets_container = QWidget()
        self.dynamic_widgets_layout = QHBoxLayout(self.dynamic_widgets_container)
        self.dynamic_widgets_layout.setContentsMargins(0, 0, 5, 0)
        view_controls_layout.addWidget(self.dynamic_widgets_container)
        
        # Initialize dynamic widget references
        self.show_masks_checkbox = None
        self.show_outlines_checkbox = None
        self.antialias_checkbox = None
        self.color_picker = None
        self.show_removal_mask_checkbox = None

        # View selector
        view_controls_layout.addWidget(QLabel("View:"))
        self.view_combo = QComboBox()
        self.view_combo.addItems([
            "Segmentation-Edit", "Segmentation-Matte", "Segmentation-BGcolor", "Matting-Matte", "Matting-BGcolor", "ObjectRemoval"
        ])

        # Always reset the view to "Segmentation-Edit"
        self.view_combo.setCurrentIndex(0)
        self.settings_mgr.set_session_setting("current_view_mode", self.view_combo.currentText())

        self.view_combo.currentTextChanged.connect(self.on_view_combo_changed)
        view_controls_layout.addWidget(self.view_combo)
        
        controls_layout.addLayout(view_controls_layout)
        layout.addLayout(controls_layout)
        
        # Initialize dynamic widgets for the default selection
        self._update_dynamic_widgets()
    
    def _on_bgcolor_changed(self, color_rgb):
        """Handle bgcolor selection"""
        self.settings_mgr.set_session_setting("bgcolor", color_rgb)
        self._update_current_frame_display()
        
    def _update_dynamic_widgets(self):
        """Update dynamic widgets (checkboxes and color picker) based on current view mode"""
        # Clear existing widgets
        self._clear_dynamic_widgets()
        
        current_view = self.view_combo.currentText()
        settings_mgr = self.settings_mgr
        
        if current_view == "Segmentation-Edit":
            # Add checkboxes for edit mode
            self.show_masks_checkbox = QCheckBox("Show masks")
            show_masks = settings_mgr.get_session_setting("show_masks", settings_mgr.app_settings.default_show_masks)
            self.show_masks_checkbox.setChecked(show_masks)
            self.show_masks_checkbox.stateChanged.connect(self.on_checkbox_changed)
            self.dynamic_widgets_layout.addWidget(self.show_masks_checkbox)
            
            self.show_outlines_checkbox = QCheckBox("Show outlines")
            show_outlines = settings_mgr.get_session_setting("show_outlines", settings_mgr.app_settings.default_show_outlines)
            self.show_outlines_checkbox.setChecked(show_outlines)
            self.show_outlines_checkbox.stateChanged.connect(self.on_checkbox_changed)
            self.dynamic_widgets_layout.addWidget(self.show_outlines_checkbox)
            
        elif current_view == "Segmentation-Matte":
            # Add antialias checkbox for matte modes
            self.antialias_checkbox = QCheckBox("Antialias")
            antialias = settings_mgr.get_session_setting("antialias", settings_mgr.app_settings.default_antialias)
            self.antialias_checkbox.setChecked(antialias)
            self.antialias_checkbox.stateChanged.connect(self.on_checkbox_changed)
            self.dynamic_widgets_layout.addWidget(self.antialias_checkbox)
            
        elif current_view == "Segmentation-BGcolor":
            # Add antialias checkbox
            self.antialias_checkbox = QCheckBox("Antialias")
            antialias = settings_mgr.get_session_setting("antialias", settings_mgr.app_settings.default_antialias)
            self.antialias_checkbox.setChecked(antialias)
            self.antialias_checkbox.stateChanged.connect(self.on_checkbox_changed)
            self.dynamic_widgets_layout.addWidget(self.antialias_checkbox)
            
            # Add color picker
            default_color = getattr(settings_mgr.app_settings, 'default_bgcolor', (0, 255, 0))
            bgcolor = settings_mgr.get_session_setting("bgcolor", default_color)
            self.color_picker = ColorPickerWidget(bgcolor)
            self.color_picker.color_changed.connect(self._on_bgcolor_changed)
            self.dynamic_widgets_layout.addWidget(self.color_picker)
            
        elif current_view == "Matting-BGcolor":
            # Add color picker
            default_color = getattr(settings_mgr.app_settings, 'default_bgcolor', (0, 255, 0))
            bgcolor = settings_mgr.get_session_setting("bgcolor", default_color)
            self.color_picker = ColorPickerWidget(bgcolor)
            self.color_picker.color_changed.connect(self._on_bgcolor_changed)
            self.dynamic_widgets_layout.addWidget(self.color_picker)

        elif current_view == "ObjectRemoval":
            # add mask overlay
            self.show_removal_mask_checkbox = QCheckBox("Show mask")
            show_removal_mask = settings_mgr.get_session_setting("show_removal_mask", settings_mgr.app_settings.default_show_removal_mask)
            self.show_removal_mask_checkbox.setChecked(show_removal_mask)
            self.show_removal_mask_checkbox.stateChanged.connect(self.on_checkbox_changed)
            self.dynamic_widgets_layout.addWidget(self.show_removal_mask_checkbox)

    def _clear_dynamic_widgets(self):
        """Remove all dynamic widgets from the container"""
        # Remove all widgets from the layout
        while self.dynamic_widgets_layout.count():
            child = self.dynamic_widgets_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Clear references
        self.show_masks_checkbox = None
        self.show_outlines_checkbox = None
        self.antialias_checkbox = None
        self.color_picker = None
        self.show_removal_mask_checkbox = None
    
    def _setup_status_bar(self):
        """Setup the status bar"""
        self.status = QLabel("Ready")
        self.statusBar = QStatusBar()
        self.statusBar.addWidget(self.status)
        self.setStatusBar(self.statusBar)
    
    def _setup_console_redirect(self):
        """Setup console redirection to the console tab"""
        self.console_redirect = ConsoleRedirect()
        self.console_redirect.text_written.connect(self.append_to_console)
        sys.stdout = self.console_redirect
        sys.stderr = self.console_redirect
        sys.excepthook = log_exception

    # ==================== EVENT HANDLERS AND CALLBACKS ====================
    
    def append_to_console(self, text, is_carriage_return=False):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.End)

        if is_carriage_return:
            # Move to start of line, replace it with new text
            cursor.movePosition(QTextCursor.StartOfLine, QTextCursor.MoveAnchor)
            cursor.movePosition(QTextCursor.EndOfLine, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(text)
        else:
            cursor.insertText(text)

        self.console.setTextCursor(cursor)
        self.console.ensureCursorVisible()
    
    def update_status_bar(self, text):
        """Update the status bar text"""
        self.status.setText(text)
    
    def on_frame_change(self, value):
        """Handle frame slider changes"""
        self.frame_value.setText(str(value))
        current_frame = value
        
        # Update current_frame in point table and refresh if show_all_points is disabled
        settings_mgr = self.settings_mgr
        show_all_points = settings_mgr.get_session_setting("show_all_points", True)
        if not show_all_points:
            self.point_table.set_current_frame(current_frame)
            self._refresh_table()
        
        view_options = self.get_view_options()
        updated_image = sammie.update_image(current_frame, view_options, self.point_manager.points)
        if updated_image:
            self.viewer.update_image(updated_image)
    
    def on_tab_changed(self, index):
        """Handle tab changes and automatically switch views"""
        # Get the tab widget to determine which tab is selected
        tab_widget = self.sidebar.tab_widget
        current_tab = tab_widget.widget(index)
        
        # Determine the appropriate view based on the current tab
        if current_tab == self.sidebar.segmentation_tab:
            # Switch to Segmentation-Edit view
            target_view = "Segmentation-Edit"
        elif current_tab == self.sidebar.matting_tab:
            # Switch to Matting-Matte view
            target_view = "Matting-Matte"
        elif current_tab == self.sidebar.removal_tab:
            # Switch to ObjectRemoval view
            target_view = "ObjectRemoval"
        else:
            # Fallback to current selection if unknown tab
            return
        
        # Only change if it's different from current selection
        current_view = self.view_combo.currentText()
        if current_view != target_view:
            # Find the index of the target view
            target_index = self.view_combo.findText(target_view)
            if target_index >= 0:
                # Change the combo box selection - this will trigger on_view_combo_changed automatically
                self.view_combo.setCurrentIndex(target_index)

    def on_view_combo_changed(self, text):
        """Handle view combobox selection changes"""
        # Save to session settings
        self.settings_mgr.set_session_setting("current_view_mode", text)
        self._update_dynamic_widgets()
        self._update_point_editing_state()
        self._update_current_frame_display()

    def on_checkbox_changed(self):
        """Handle checkbox state changes"""
        settings_mgr = self.settings_mgr
        
        # Update session settings
        if self.show_masks_checkbox:
            settings_mgr.set_session_setting("show_masks", self.show_masks_checkbox.isChecked())
        if self.show_outlines_checkbox:
            settings_mgr.set_session_setting("show_outlines", self.show_outlines_checkbox.isChecked())
        if self.antialias_checkbox:
            settings_mgr.set_session_setting("antialias", self.antialias_checkbox.isChecked())
        if self.show_removal_mask_checkbox:
            settings_mgr.set_session_setting("show_removal_mask", self.show_removal_mask_checkbox.isChecked())
        
        # Update the display
        self._update_current_frame_display()

    def _on_points_changed(self, action, **kwargs):
        """Update GUI when point data changes"""
        if action == 'add':
            point = kwargs['point']
            # Ensure current_frame is up to date before adding point
            self.point_table.set_current_frame(self.frame_slider.value())
            self.point_table.add_point(point['frame'], point['object_id'], point['positive'], point['x'], point['y'])
            # No image update here - wait for segmentation to complete
            
        elif action == 'remove_last':
            point = kwargs.get('point')
            if point:
                self.point_table.remove_last_point()
                points = self.point_manager.get_all_points()
                self.sam_manager.clear_tracking()
                self.matany_manager.propagated = False
                self.sam_manager.replay_points(points)
                # No image update here - wait for segmentation to complete
        
        elif action == 'clear_frame':
            points = self.point_manager.get_all_points()
            self.sam_manager.clear_tracking()
            self.matany_manager.propagated = False
            self.sam_manager.replay_points(points)
            # No image update here - wait for segmentation to complete
            self._refresh_table()
            
        elif action == 'clear_all':
            self.sam_manager.clear_tracking()
            self.matany_manager.propagated = False
            # Rebuild the entire table and update display
            self._refresh_table()
            self._update_current_frame_display()
        
        elif action == 'clear_object':
            object_id = kwargs.get('object_id')
            if object_id is not None:
                # Call predictor remove_object
                self.sam_manager.predictor.remove_object(self.sam_manager.inference_state, object_id)
            if len(self.point_manager.points) == 0:
                # Reset predictor when no points remain
                self.sam_manager.predictor.reset_state(self.sam_manager.inference_state)
                self.sam_manager.propagated = False
                self.matany_manager.propagated = False
            # Rebuild the entire table and update display
            self._refresh_table()
            self._update_current_frame_display()
            
        elif action == 'load_all':
            # Rebuild the entire table and update display
            self._refresh_table()
            self._update_current_frame_display()
        self.update_tracking_status()
        self.update_matting_status()
        self.update_removal_status()
    
    def _on_point_selected(self, point_data):
        """Handle point selection from table"""
        if point_data:
            # Store the highlighted point
            self.highlighted_point = point_data
            # Navigate to the frame (incase of multiple points, the frame of the last point in the list)
            target_frame = point_data[-1]['frame']
            # Only update if frame is different to avoid unnecessary updates
            if self.frame_slider.value() != target_frame:
                self.frame_slider.setValue(target_frame)
            else:
                self._update_current_frame_display()
                
        else:
            # Clear highlight
            self.highlighted_point = None
            self._update_current_frame_display()

    def _on_object_name_changed(self):
        """Refresh point table when object names change"""
        # Store current scroll position
        if hasattr(self, 'point_table'):
            scroll_value = self.point_table.verticalScrollBar().value()
            
            # Rebuild table to show updated names
            self._refresh_table()

            # Resize the Object ID column (column 1)
            self.point_table.resizeColumnToContents(1)
            
            # Restore scroll position
            self.point_table.verticalScrollBar().setValue(scroll_value)
            
    def _on_segmentation_changed(self, action, **kwargs):
        """Handle segmentation events and update display"""
        if action == 'segmentation_complete':
            frame = kwargs.get('frame')
            current_frame = self.frame_slider.value()
            
            # Only update display if this segmentation is for the current frame
            if frame == current_frame:
                self._update_current_frame_display()
                
        elif action == 'replay_complete':
            # Update display after replay is complete
            self._update_current_frame_display()

    def _update_current_frame_display(self):
        """Update the current frame display with masks and points"""
        if sammie.VideoInfo.total_frames == 0:
            return  # Don't try to update before video is loaded
        current_frame = self.frame_slider.value()
        view_options = self.get_view_options()
        updated_image = sammie.update_image(current_frame, view_options, self.point_manager.points)
        if updated_image:
            self.viewer.update_image(updated_image)
            
    def _refresh_table(self):
        """Rebuild table from point manager data"""
        settings_mgr = self.settings_mgr
        show_all_points = settings_mgr.get_session_setting("show_all_points", True)
        current_frame = self.frame_slider.value()
        
        # Update current_frame in point table
        self.point_table.set_current_frame(current_frame)
        
        self.point_table.clear_points()
        for point in self.point_manager.points:
            # If show_all_points is disabled, only add points for current frame
            if not show_all_points and point['frame'] == current_frame:
                self.point_table.add_point(point['frame'], point['object_id'], point['positive'], point['x'], point['y'])
            elif show_all_points:
                self.point_table.add_point(point['frame'], point['object_id'], point['positive'], point['x'], point['y'])

    def closeEvent(self, event):
        """Clean up on close"""
        # Save current session state
        self._save_current_ui_state()
        self.settings_mgr.save_points(self.point_manager.get_all_points())
        self.settings_mgr.save_session_settings()
        
        self._save_window_and_splitter_settings()

        # Clean up console redirect
        if hasattr(self, 'console_redirect'):
            self.console_redirect.close()
        event.accept()

    # ==================== POINT OPERATIONS ====================
    
    def add_point_from_click(self, x, y, is_positive):
        """Add point when user clicks image"""
        current_frame = self.frame_slider.value()
        seg_tab = self.sidebar.segmentation_tab
        
        object_id = seg_tab.get_selected_object_id()
        
        # Determine point type from the click
        point_type = "positive" if is_positive else "negative"
        
        # Add point to table
        if self.sam_manager.propagated or self.matany_manager.propagated:
            self.clear_tracking_data() #clear tracking and replay points
            self._add_point_and_segment(current_frame, object_id, is_positive, x, y, point_type)
        else:
            self._add_point_and_segment(current_frame, object_id, is_positive, x, y, point_type)

    def _add_point_and_segment(self, frame, object_id, is_positive, x, y, point_type):
        """Helper method to add point and trigger segmentation"""
        # Add the point (this will trigger point callback but not update image yet)
        self.point_manager.add_point(frame, object_id, is_positive, x, y)
        print(f"Added point: Frame {frame}, Object {object_id}, "
              f"Type: {point_type}, Position: ({x}, {y})")
        
        # Get points for segmentation
        coordinates, labels = self.point_manager.get_sam2_points(frame, object_id)
        
        # Run segmentation (this will trigger segmentation callback and update image)
        self.sam_manager.segment_image(frame, object_id, coordinates, labels)  
    
    def delete_selected_point(self):
        """Delete the currently selected point in the table"""
        if hasattr(self, 'point_table'):
            row = self.point_table.selectionModel().currentIndex().row()
            self.point_table.delete_selected_row(single_row=row)

    def replay_all_points(self):
        """Replay all points to rebuild masks"""
        if not self.point_manager.points:
            print("No points to replay")
            return
        
        print(f"Replaying {len(self.point_manager.points)} points...")
        self.sam_manager.replay_points(self.point_manager.get_all_points())
        
        # Update display after replay
        self._update_current_frame_display()
    
    def undo_last_point(self):
        """Remove last point"""
        removed_point = self.point_manager.remove_last()
        frame, object_id, point_type, x, y = removed_point.values()
        if removed_point:
            print(f"Deleted point: Frame {frame}, Object {object_id}, "
              f"Type: {point_type}, Position: ({x}, {y})")
        else:
            print("No points to remove")

    def clear_object_points(self):
        """Clear points for selected object"""
        seg_tab = self.sidebar.segmentation_tab
        object_id = seg_tab.get_selected_object_id()
        
        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete all points for object {object_id}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return  # Exit early if canceled

        removed_count = self.point_manager.clear_object(object_id)
        if removed_count > 0:
            print(f"Cleared {removed_count} points for object {object_id}")
        else:
            print(f"No points found for object {object_id}")

    def clear_frame_points(self):
        """Clear points for current frame"""
        current_frame = self.frame_slider.value()

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete all points on this frame?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return  # Exit early if canceled
        
        removed_count = self.point_manager.clear_frame(current_frame)
        if removed_count > 0:
            print(f"Cleared {removed_count} points for frame {current_frame}")
        else:
            print(f"No points found for frame {current_frame}")
    
    def clear_all_points(self):
        """Clear all points"""
        count = len(self.point_manager.points)

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete ALL points in this project?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return  # Exit early if canceled
        
        self.point_manager.clear_all()
        if count > 0:
            print(f"Cleared all {count} points")
        else:
            print("No points to clear")

    def get_sam2_points(self, object_id=None):
        """Get points for current frame in SAM2 format"""
        current_frame = self.frame_slider.value()
        return self.point_manager.get_sam2_points(current_frame, object_id)
    
    def get_frame_points(self, frame=None):
        """Get points for a specific frame (or current frame)"""
        if frame is None:
            frame = self.frame_slider.value()
        return self.point_manager.get_points_for_frame(frame)

    # ==================== PROCESSING OPERATIONS ====================
    
    def track_objects(self):
        """Run object tracking using current points"""
        self.settings_mgr.save_session_settings()
        count = len(self.point_manager.points)
        if count > 0:            
            self.sam_manager.clear_tracking()
            self.sam_manager.replay_points(self.point_manager.get_all_points())
            if self.sam_manager.track_objects(parent_window=self) == 0: # if cancelled
                self.sam_manager.clear_tracking() 
                self.sam_manager.replay_points(self.point_manager.get_all_points())
            else: # completed
                pass
            sammie.remove_backup_mattes() # Make sure to remove an existing mattes backup folder
            self.update_tracking_status()
            self._update_current_frame_display()
            self.settings_mgr.save_session_settings()
        else:
            # If no points, just update display
            print("Points must be added before tracking")
            self._update_current_frame_display()
    
    def clear_tracking_data(self):
        """Clear tracking data"""
        self.sam_manager.clear_tracking()  # This already handles the clearing both propagated and deduplicated
        self.matany_manager.propagated = False # Clear matting flag
        self.sam_manager.deduplicated = False # Clear deduplication flag
        sammie.remove_backup_mattes() # Make sure to remove an existing mattes backup folder
        self.update_tracking_status()
        self.update_matting_status()
        self.update_removal_status()
        # Replay points to rebuild masks if there are any points
        if self.point_manager.points:
            self.sam_manager.replay_points(self.point_manager.get_all_points())
        else:
            # If no points, just update display
            self._update_current_frame_display()
    
    def deduplicate_similar_masks(self):
        """Deduplicate similar masks"""
        if not self.sam_manager.propagated:
            print("Run Track Objects before Deduplication")
            return
        #print("Running deduplication...")
        self.sam_manager.deduplicated = sammie.deduplicate_masks(parent_window=self)
        
        # Update display after deduping finished
        self._update_current_frame_display()
        
        # Update the button status
        if hasattr(self, 'segmentation_tab'):
            self.update_deduplicate_status()
        self.settings_mgr.save_session_settings()

    def run_matting(self):
        """Run matting process"""
        self.settings_mgr.save_session_settings()
        count = len(self.point_manager.points)
        if count > 0:  
            #load models
            print("Loading MatAnyone model...")
            QApplication.processEvents()
            self.sam_manager.offload_model_to_cpu()
            #self.matany_manager.load_model_to_device()
            self.matany_manager.load_matting_model() # load matting model for first time
            QApplication.processEvents()
            success = self.matany_manager.run_matting(self.point_manager.points, parent_window=self)
        
            if success:
                self.matany_manager.propagated = True
                self.frame_slider.setValue(0)
                self.update_matting_status()
                self._update_current_frame_display()
                
            else:
                self.matany_manager.propagated = False
                self.update_matting_status()
                self._update_current_frame_display()
            # self.matany_manager.offload_model_to_cpu()
            QApplication.processEvents()
            self.settings_mgr.save_session_settings()
            self.matany_manager.unload_matting_model() # unload matting model since its not needed in memory
            self.sam_manager.load_model_to_device()
        else:
            print("Points must be added on the Segmentation tab before matting")

    def run_object_removal(self):
        """Run object removal process"""

        # Don't allow minimax-remover on CPU
        if sammie.DeviceManager.get_device().type == 'cpu' and self.removal_tab.method_combo.currentText() == 'MiniMax-Remover':
            QMessageBox.warning(self, "Error", "MiniMax-Remover is not supported on CPU. Please use OpenCV instead.", QMessageBox.Ok)
            return
        self.settings_mgr.save_session_settings()
        if self.sam_manager.propagated:
            # offload sam model
            self.sam_manager.offload_model_to_cpu()

            if self.removal_tab.method_combo.currentText() == 'MiniMax-Remover':
                success = False
                try:
                    success = self.removal_manager.run_object_removal_minimax(self.point_manager.points, parent_window=self)
                except Exception as e:
                    if "out of memory" in str(e):
                        print(e)
                        QMessageBox.warning(self, "Error", "An out of memory error occurred. Please restart the application to fully release GPU memory, and try again with lower settings.", QMessageBox.Ok)
                    else: 
                        print(f"Error running MiniMax-Remover: {e}")
            else:
                success = self.removal_manager.run_object_removal_cv(self.point_manager.points, parent_window=self)
            if success:
                self.removal_manager.propagated = True
                self.frame_slider.setValue(0)
                self.update_removal_status()
                self._update_current_frame_display()
            else:
                self.removal_manager.propagated = False
                self.update_removal_status()
                self._update_current_frame_display()

            self.settings_mgr.save_session_settings()
            # unload minimax model and load sam model
            self.removal_manager.unload_minimax_model()    
            self.sam_manager.load_model_to_device()
        else:
            print("You must track objects on the Segmentation tab before object removal")

    def update_tracking_status(self):
        """Update the tracking status display"""
        # Update the button status
        if hasattr(self, 'segmentation_tab'):
            self.segmentation_tab.update_tracking_status_helper(self.sam_manager.propagated)
            # Clear deduplication, matting, and dedupe status when tracking is cleared
            if not self.sam_manager.propagated:
                self.sam_manager.deduplicated = False
                self.matany_manager.propagated = False
                self.removal_manager.propagated = False
            self.update_deduplicate_status()

    def update_deduplicate_status(self):
        """Update the Deduplicate button text based on deduplication status"""
        self.segmentation_tab.update_deduplicate_status_helper(self.sam_manager.deduplicated)

    def update_matting_status(self):
        """Update the matting status display"""
        if hasattr(self, 'matting_tab'):
            self.matting_tab.update_matting_status(self.matany_manager.propagated)

    def update_removal_status(self):
        """Update the removal status display"""
        if hasattr(self, 'removal_tab'):
            self.removal_tab.update_removal_status(self.removal_manager.propagated)
            if self.show_removal_mask_checkbox is not None:
                self.show_removal_mask_checkbox.setChecked(not self.removal_manager.propagated)

    # ==================== PLAYBACK CONTROLS ====================
    
    def toggle_play_pause(self):
        """Toggle between play and pause states"""
        if not self.is_playing:
            # If we are on the last frame, go to the beginning
            if self.frame_slider.value() == self.frame_slider.maximum():
                self.frame_slider.setValue(0)
            # Start playback
            self.is_playing = True
            self.play_pause_btn.setIcon(self.icon_pause)
            # Adjust interval to your desired FPS (e.g. ~33 ms for 30fps, 40 ms for 25fps)
            self.play_timer.start(40)

        else:
            # Pause playback
            self.is_playing = False
            self.play_pause_btn.setIcon(self.icon_play)
            self.play_timer.stop()
    
    def play_next_frame(self):
        """Advance to the next frame during playback"""
        current_value = self.frame_slider.value()
        if current_value < self.frame_slider.maximum():
            self.frame_slider.setValue(current_value + 1)
        else:
            # Stop at the end
            self.toggle_play_pause()
        
    def prev_frame(self):
        """Go to previous frame"""
        current_value = self.frame_slider.value()
        if current_value > 0:
            self.frame_slider.setValue(current_value - 1)
    
    def next_frame(self):
        """Go to next frame"""
        current_value = self.frame_slider.value()
        if current_value < self.frame_slider.maximum():
            self.frame_slider.setValue(current_value + 1)
    
    def prev_keyframe(self):
        """Go to previous frame that contains points"""
        current_frame = self.frame_slider.value()
        
        # Get all frames that have points, sorted in descending order
        frames_with_points = sorted(set(point['frame'] for point in self.point_manager.points), reverse=True)
        
        # Find the previous frame with points
        prev_frame = None
        for frame in frames_with_points:
            if frame < current_frame:
                prev_frame = frame
                break
        
        if prev_frame is not None:
            self.frame_slider.setValue(prev_frame)
            print(f"Previous point frame: {prev_frame}")
        else:
            self.goto_first_frame()
    
    def next_keyframe(self):
        """Go to next frame that contains points"""
        current_frame = self.frame_slider.value()
        
        # Get all frames that have points, sorted in ascending order
        frames_with_points = sorted(set(point['frame'] for point in self.point_manager.points))
        
        # Find the next frame with points
        next_frame = None
        for frame in frames_with_points:
            if frame > current_frame:
                next_frame = frame
                break
        
        if next_frame is not None:
            self.frame_slider.setValue(next_frame)
            print(f"Next point frame: {next_frame}")
        else:
            self.goto_last_frame()
    
    def goto_first_frame(self):
        """Go to the first frame"""
        self.frame_slider.setValue(0)
    
    def goto_last_frame(self):
        """Go to the last frame"""
        self.frame_slider.setValue(self.frame_slider.maximum())

    # ==================== VIEW CONTROLS ====================
    
    def set_view_mode(self, mode):
        """Set the view mode, used for hotkeys"""
        index = self.view_combo.findText(mode)
        if index >= 0:
            self.view_combo.setCurrentIndex(index)
            print(f"Switched to {mode}")

    def _update_point_editing_state(self):
        """Enable/disable point editing based on current view"""
        is_edit_view = self.view_combo.currentText() == "Segmentation-Edit"
        
        # Enable/disable point clicks in image viewer (but keep zoom/pan)
        self.viewer.point_editing_enabled = is_edit_view
    
    def set_object_id(self, object_id):
        """Set the selected object ID in the segmentation tab"""
        if hasattr(self.sidebar, 'segmentation_tab'):
            # Clamp to valid range
            max_id = self.sidebar.segmentation_tab.object_spinbox.maximum()
            object_id = min(object_id, max_id)
            self.sidebar.segmentation_tab.object_spinbox.setValue(object_id)
            print(f"Selected object {object_id}")
    
    def get_view_options(self):
        """Get current view options from settings"""
        view_options = self.settings_mgr.get_view_options()
        # Add the currently highlighted point (transient UI state, not saved)
        view_options['highlighted_point'] = getattr(self, 'highlighted_point', None)
        return view_options
    
    def reset_interface(self):
        """Reset the interface to default panel sizes"""
        # Use the original default values from ApplicationSettings dataclass
        defaults = ApplicationSettings()
        
        self.main_splitter.setSizes(defaults.main_splitter_sizes)
        self.vertical_splitter.setSizes(defaults.vertical_splitter_sizes)
        self.bottom_splitter.setSizes(defaults.bottom_splitter_sizes)
        print("Interface reset to default layout")
    
    def fit_to_screen(self):
        """Fit image to screen"""
        if self.viewer:
            self.viewer.zoom_to_fit()
    
    def zoom_100(self):
        """Set zoom to 100%"""
        if self.viewer:
            self.viewer.zoom_to_100()
    
    def zoom_200(self):
        """Set zoom to 200%"""
        if self.viewer:
            self.viewer.set_zoom(2.0)
    
    def zoom_400(self):
        """Set zoom to 400%"""
        if self.viewer:
            self.viewer.set_zoom(4.0)
    
    def zoom_in(self):
        """Zoom in by 1.5x"""
        if self.viewer:
            current_scale = self.viewer.current_scale
            self.viewer.set_zoom(current_scale * 1.5)
    
    def zoom_out(self):
        """Zoom out by 1/1.5x"""
        if self.viewer:
            current_scale = self.viewer.current_scale
            self.viewer.set_zoom(current_scale / 1.5)

    # ==================== KEYBOARD SHORTCUTS ====================

    def _setup_hotkeys(self):
        """Setup keyboard shortcuts for common actions"""
        # DONT create shortcuts for items that don't appear in the menu
        
        # File operations
        self._create_shortcut("Ctrl+O", self.open_file, "Load Video", create_shortcut=False)
        self._create_shortcut("Ctrl+L", self.load_points, "Load Points", create_shortcut=False)
        self._create_shortcut("Ctrl+Shift+L", self.load_project, "Load Project", create_shortcut=False)
        self._create_shortcut("Ctrl+S", self.save_points, "Save Points", create_shortcut=False)
        self._create_shortcut("Ctrl+Shift+S", self.save_project, "Save Project", create_shortcut=False)
        self._create_shortcut("Ctrl+E", self.export_video, "Export Video", create_shortcut=False)
        self._create_shortcut("Ctrl+Shift+E", self.export_image, "Export Image", create_shortcut=False)
        
        # View/Zoom controls
        self._create_shortcut("Backspace", self.zoom_100, "100% Zoom", create_shortcut=False)
        self._create_shortcut("Ctrl+Backspace", self.fit_to_screen, "Fit to Screen", create_shortcut=False)
        self._create_shortcut("=", self.zoom_in, "Zoom In")
        self._create_shortcut("-", self.zoom_out, "Zoom Out")
        self._create_shortcut("Ctrl+Shift+R", self.reset_interface, "Reset Interface", create_shortcut=False)
        
        # Frame navigation
        self._create_shortcut(",", self.prev_frame, "Previous Frame")
        self._create_shortcut(".", self.next_frame, "Next Frame")
        self._create_shortcut("PgUp", self.prev_keyframe, "Previous Keyframe")
        self._create_shortcut("PgDown", self.next_keyframe, "Next Keyframe")
        self._create_shortcut("Home", self.goto_first_frame, "Go to First Frame")
        self._create_shortcut("End", self.goto_last_frame, "Go to Last Frame")
        self._create_shortcut("Space", self.toggle_play_pause, "Play/Pause")
        
        # Point operations
        self._create_shortcut("Ctrl+Z", self.undo_last_point, "Remove Last Point")
        self._create_shortcut("Delete", self.delete_selected_point, "Delete Selected Point")
        self._create_shortcut("Ctrl+Delete", self.clear_frame_points, "Clear Frame Points")
        self._create_shortcut("Shift+Delete", self.clear_object_points, "Clear Object Points")
        self._create_shortcut("Ctrl+Shift+Delete", self.clear_all_points, "Clear All Points")
        
        # Processing operations
        self._create_shortcut("Ctrl+T", self.track_objects, "Track Objects")
        self._create_shortcut("Ctrl+M", self.run_matting, "Run Matting")
        self._create_shortcut("Ctrl+D", self.deduplicate_similar_masks, "Deduplicate Masks")
        self._create_shortcut("Ctrl+R", self.clear_tracking_data, "Clear Tracking Data")

        # Object selection (number keys 0-9 for quick object switching)
        for i in range(10):
            self._create_shortcut(str(i), lambda obj_id=i: self.set_object_id(obj_id), f"Select Object {i}")
        
        # View mode switching
        self._create_shortcut("F2", lambda: self.set_view_mode("Segmentation-Edit"), "Segmentation Edit View")
        self._create_shortcut("F3", lambda: self.set_view_mode("Segmentation-Matte"), "Segmentation Matte View")
        self._create_shortcut("F4", lambda: self.set_view_mode("Segmentation-BGcolor"), "Segmentation BGcolor View")
        self._create_shortcut("F5", lambda: self.set_view_mode("Matting-Matte"), "Matting Matte View")
        self._create_shortcut("F6", lambda: self.set_view_mode("Matting-BGcolor"), "Matting BGcolor View")
        self._create_shortcut("F7", lambda: self.set_view_mode("ObjectRemoval"), "Object Removal Edit View")
        
        # Help
        self._create_shortcut("F1", self.show_help, "Show Help", create_shortcut=False)
        self._create_shortcut("Ctrl+F1", self.show_hotkeys_help, "Show Keyboard Shortcuts", create_shortcut=False)

    def _create_shortcut(self, key, action, description, create_shortcut=True):
        """Helper method to create keyboard shortcuts"""
        if create_shortcut:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
        
        # Always add to the list for help display
        if not hasattr(self, "_shortcuts_list"):
            self._shortcuts_list = []
        self._shortcuts_list.append((key, description))

    def show_hotkeys_help(self):
        if hasattr(self, "_shortcuts_list"):
            dlg = HotkeysHelpDialog(self._shortcuts_list, self)
            dlg.exec()

    # ==================== FILE OPERATIONS ====================
    
    def open_file(self):
        """Open an image file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, 
            "Open File", 
            "", 
            "*.mp4 *.m4v *.mkv *.mov *.avi *webm *.png *.jpg *.jpeg *.bmp *.tiff *.gif *.webp"
        )
        
        if file_name:  # Only proceed if a file was selected
            success = self.load_file(file_name)
            if not success:
                print("Failed to load file or loading was cancelled.")
                # Reset UI to empty state on failure
                self.frame_slider.setRange(0, 0)
                self.frame_slider.setValue(0)
                self.viewer.clear_image()
    
    def handle_dropped_file(self, file_path):
        """Handle file dropped onto the viewer"""
        if not file_path or not os.path.exists(file_path):
            print(f"Invalid file path: {file_path}")
            return
        
        # Check if file type is supported
        supported_extensions = [
            '.mp4', '.m4v', '.mkv', '.mov', '.avi', '.webm',
            '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'
        ]
        
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in supported_extensions:
            print(f"Unsupported file type: {file_ext}")
            QMessageBox.warning(
                self,
                "Unsupported File",
                f"File type '{file_ext}' is not supported.\n\n"
                f"Supported formats: {', '.join(supported_extensions)}"
            )
            return
        
        self.load_file(file_path)
        
    def load_file(self, file_path):
        """Load a file (consolidated method for both menu and command line usage)"""
        # Clear all points and propagation data
        self.point_manager.clear_all()
        self.sam_manager.propagated = False
        self.matany_manager.propagated = False

        # Create new session
        self.settings_mgr.create_new_session(file_path)
        
        # Reset UI
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self._reset_show_all_points_button_state()
        self.viewer.clear_image()
        self.sidebar.load_values_from_settings()
        self.sidebar.tab_widget.setCurrentIndex(0)
        self._update_dynamic_widgets()
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp']:
                framecount = sammie.load_image_sequence(file_path, parent_window=self)
            else:
                framecount = sammie.load_video(file_path, parent_window=self)
                
            if framecount and framecount > 0:
                # Save video info to session
                video_info = sammie.VideoInfo
                self.settings_mgr.update_video_info(
                    video_info.width, video_info.height, 
                    video_info.fps, video_info.total_frames, file_path
                )
                
                # If png or jpg was loaded, set the frame format to override the app setting
                if file_ext in ['.png', '.jpg', '.jpeg']:
                    frame_format = file_ext.lstrip('.')
                    if frame_format == 'jpeg':
                        frame_format = 'jpg'  # Normalize jpeg to jpg
                    self.settings_mgr.set_session_setting("frame_format", frame_format)
                    
                self.settings_mgr.save_session_settings()
                
                print(f"Loaded {framecount} frames")
                
                # Initialize the predictor
                self.sam_manager.initialize_predictor()
                
                # Update frame slider range
                self.frame_slider.setRange(0, framecount-1)
                
                # Load and display the first frame - reset zoom for new video
                current_frame = 0
                view_options = self.get_view_options()
                updated_image = sammie.update_image(current_frame, view_options, self.point_manager.points)
                if updated_image:
                    self.viewer.load_image_reset_zoom(updated_image)  # Reset zoom for new content
                    self.fit_to_screen()
                self.frame_slider.setValue(0)
                return True
            else:
                print(f"Failed to load file: {file_path}")
                return False
                
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return False

    def resume_prev_session(self):
        """Resume previous session"""
        # Session settings loaded earlier in __init__, just check if session exists
        session_loaded = self.settings_mgr.session_exists() 
        framecount = sammie.resume_session()
        if framecount is not None and framecount > 0:
            print(f"Loaded {framecount} frames")
            # initialize the predictor
            self.sam_manager.initialize_predictor()
            # Update frame slider range
            self.frame_slider.setRange(0, framecount-1)
            # Load points if session was loaded
            if session_loaded:
                points = self.settings_mgr.load_points()
                if points:
                    self.point_manager.points = points
                    self.point_manager._notify('load_all')
                
                # Load UI state from session
                self._load_session_ui_state()

            # Load and display the first frame - reset zoom for resumed session
            current_frame = 0
            view_options = self.get_view_options()
            updated_image = sammie.update_image(current_frame, view_options, self.point_manager.points)
            if updated_image:
                self.viewer.load_image_reset_zoom(updated_image)  # Reset zoom for resumed session
            self.frame_slider.setValue(0)
    
    def save_points(self):
        """Save points to file"""
        self.settings_mgr.save_session_settings()

        save_dialog = QFileDialog(self, "Save Points", "")
        save_dialog.setAcceptMode(QFileDialog.AcceptSave)
        save_dialog.setNameFilter("JSON Files (*.json)")
        save_dialog.setDefaultSuffix("json")

        # Wait for the user to execute the save before continuing.
        if save_dialog.exec_() == QDialog.Accepted:
            file_name = save_dialog.selectedFiles()[0]
            if file_name:
                points = self.point_manager.get_all_points()
                with open(file_name, 'w') as f:
                    json.dump(points, f, indent=2)
                print(f"Saved {len(points)} points to {file_name}")

    def load_points(self):
        """Load points from file"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Points", "", "JSON Files (*.json)")
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    points = json.load(f)
                
                self.point_manager.points = points
                self.point_manager._notify('load_all')
                print(f"Loaded {len(points)} points from {file_name}")
                self.clear_tracking_data() #clear tracking and replay points
            except Exception as e:
                print(f"Error loading points: {e}")

    def save_project(self):
        """Save project to file"""
        self.settings_mgr.save_session_settings()

        save_dialog = QFileDialog(self, "Save Project", "")
        save_dialog.setAcceptMode(QFileDialog.AcceptSave)
        save_dialog.setNameFilter("Sammie Files (*.sammie)")
        save_dialog.setDefaultSuffix("sammie")

        # Wait for the user to execute the save before continuing.
        if save_dialog.exec_() == QDialog.Accepted:
            file_name = save_dialog.selectedFiles()[0]
            if file_name:
                success = sammie.save_project(file_name, parent_window=self)
                if success:
                    print(f"Saved project to {file_name}")
                else:
                    print("Failed to save project")
    
    def load_project(self):
        """Load project from file"""
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Sammie Files (*.sammie)")
        if file_name:
            success = sammie.load_project(file_name, parent_window=self)
            if success: 
                print(f"Loaded project from {file_name}")
                self.settings_mgr.load_session_settings()
                self.resume_prev_session()
                return
            else:
                print("Failed to load project")
                return

    def export_video(self):
        """Open export dialog"""
        self.settings_mgr.save_session_settings()
        self.settings_mgr.save_points(self.point_manager.get_all_points())
        if sammie.VideoInfo.total_frames == 0:
            QMessageBox.warning(self, "Export Error", 
                              "No video data available. Please load a video first.")
            return
        
        dialog = ExportDialog(self)
        dialog.exec()
    
    def export_image(self):
        """Open export image dialog"""
        self.settings_mgr.save_session_settings()
        self.settings_mgr.save_points(self.point_manager.get_all_points())
        if sammie.VideoInfo.total_frames == 0:
            QMessageBox.warning(self, "Export Error", 
                              "No video data available. Please load a video first.")
            return
        frame = self.frame_slider.value()
        dialog = ImageExportDialog(self, frame)
        dialog.exec()

    # ==================== SETTINGS AND STATE ====================

    def _load_session_ui_state(self, reset_view=True):
        """Load UI state from session settings"""
        settings_mgr = self.settings_mgr
        
        # Load view mode
        if reset_view:
            # Always reset the view to "Segmentation-Edit"
            self.view_combo.setCurrentIndex(0)
            self.settings_mgr.set_session_setting("current_view_mode", self.view_combo.currentText())

        # Load sidebar values
        self.sidebar.load_values_from_settings()
        
        # Update checkboxes
        self._update_dynamic_widgets()
        
        # Load tracking state
        self.sam_manager.propagated = settings_mgr.get_session_setting("is_propagated", False)
        self.sam_manager.deduplicated = settings_mgr.get_session_setting("is_deduplicated", False)
        self.matany_manager.propagated = settings_mgr.get_session_setting("is_matted", False)
        self.removal_manager.propagated = settings_mgr.get_session_setting("is_removed", False)
        self.update_tracking_status() # also updates deduplication status
        self.update_matting_status()
        self.update_removal_status()
        # Update display
        self._update_current_frame_display()

    def _save_window_and_splitter_settings(self):
        """Save current window size, state, and all splitter sizes to application settings"""
        # Save window size (only if not maximized, to preserve restored size)
        if not self.isMaximized():
            size = self.size()
            self.settings_mgr.set_app_setting("window_width", size.width())
            self.settings_mgr.set_app_setting("window_height", size.height())
        
        # Save maximized state
        self.settings_mgr.set_app_setting("window_maximized", self.isMaximized())
        
        # Save all splitter sizes
        self.settings_mgr.set_app_setting("main_splitter_sizes", self.main_splitter.sizes())
        self.settings_mgr.set_app_setting("vertical_splitter_sizes", self.vertical_splitter.sizes())
        self.settings_mgr.set_app_setting("bottom_splitter_sizes", self.bottom_splitter.sizes())
        
        # Save to disk
        self.settings_mgr.save_app_settings()

    def show_settings(self):
        """Show settings dialog"""
        self.settings_mgr.save_app_settings()
        self.settings_mgr.save_session_settings()
        # get model settings
        cpu = self.settings_mgr.get_app_setting("force_cpu", 0)
        segmentation = self.settings_mgr.get_app_setting("sam_model", "None")
        dialog = SettingsDialog(self.settings_mgr, self)
        if dialog.exec():
            print("Application settings saved.")

            # Prompt to restart if needed
            if (cpu != self.settings_mgr.get_app_setting("force_cpu", 0)
                or segmentation != self.settings_mgr.get_app_setting("sam_model", "None")):
                QMessageBox.information(self, "Restart Required", "You must restart the application for model or device changes to take effect.")

    # ==================== UPDATE CHECKER ====================
    
    def on_update_available(self, current_version, latest_version):
        """Handle update available signal from background thread"""
        # Print to console
        print(f"🔔 A new version of Sammie-roto is available! ({latest_version}) It can be downloaded from the File menu.")
        
        # Add menu item to file menu if it doesn't already exist
        if self.update_menu_action is None:
            # Add separator before update item
            self.file_menu.addSeparator()
            
            # Create update menu action
            self.update_menu_action = QAction(f"Update Available ({latest_version})", self)
            self.update_menu_action.triggered.connect(
                lambda: self.open_update_url(latest_version)
            )
            
            # Find the Exit action and insert before it
            actions = self.file_menu.actions()
            exit_action = None
            
            # Look for the Exit action
            for action in actions:
                if action.text() == "Exit":
                    exit_action = action
                    break
            
            if exit_action:
                # Insert separator before Exit if there isn't one already
                separator_before_exit = False
                exit_index = actions.index(exit_action)
                if exit_index > 0 and actions[exit_index - 1].isSeparator():
                    separator_before_exit = True
                
                if not separator_before_exit:
                    self.file_menu.insertSeparator(exit_action)
                
                # Insert update action before Exit
                self.file_menu.insertAction(exit_action, self.update_menu_action)
            else:
                # Fallback: add at the end if Exit not found
                self.file_menu.addSeparator()
                self.file_menu.addAction(self.update_menu_action)
    
    def open_update_url(self, version):
        """Open the GitHub releases page"""
        url = "https://github.com/Zarxrax/Sammie-Roto-2/releases"
        webbrowser.open(url)
    
    def show_help(self):
        """Open the GitHub wiki page"""
        url = "https://github.com/Zarxrax/Sammie-Roto-2/wiki"
        webbrowser.open(url)

    def show_about(self):
        """Show about dialog"""
        msg = QMessageBox(self)
        msg.setWindowTitle("About Sammie-Roto")
        msg.setText(f"Sammie-Roto Version {__version__}")
        
        # Use rich text to make the URL clickable
        info_text = (
            "Video Segmentation and Matting tool<br><br>"
            '<a href="https://github.com/Zarxrax/Sammie-Roto-2">https://github.com/Zarxrax/Sammie-Roto-2</a>'
        )
        msg.setInformativeText(info_text)
        msg.setTextFormat(Qt.RichText)
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()


# ==================== APPLICATION ENTRY POINT ====================

def main():
    """Main application entry point"""
    # Parse command line arguments when called directly
    parser = argparse.ArgumentParser(
        description="Sammie-Roto: Video Segmentation and Matting Tool",
        add_help=False  # We'll add help manually to avoid conflicts
    )
    parser.add_argument('file', nargs='?', help='Path to video or image file to load')
    parser.add_argument('--file', '-f', dest='file_flag', help='Path to video or image file to load')
    parser.add_argument('--help', '-h', action='store_true', help='Show this help message')
    
    args = parser.parse_args()
    
    if args.help:
        parser.print_help()
        print("""
Examples:
  python sammie_main.py                       # Start with GUI file picker
  python sammie_main.py video.mp4            # Load specific video file
  python sammie_main.py image.jpg            # Load specific image file
  python sammie_main.py --file video.mp4     # Load using --file flag
        """)
        return
    
    # Use either positional argument or --file flag
    file_to_load = args.file or args.file_flag
    
    # Validate file exists if provided
    if file_to_load:
        if not os.path.exists(file_to_load):
            print(f"Error: File not found: {file_to_load}")
            return

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(":/icon.ico"))

    window = MainWindow(initial_file=file_to_load)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
