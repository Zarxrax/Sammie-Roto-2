# sammie/image_export_dialog.py
import os
from PIL import Image
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QComboBox, QCheckBox,
    QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from sammie import sammie


class ImageExportDialog(QDialog):
    """Dialog for exporting a single frame as an image"""
    
    def __init__(self, parent=None, frame_number=0):
        super().__init__(parent)
        self.parent_window = parent
        self.frame_number = frame_number
        self.setWindowTitle("Export Image")
        self.setModal(True)
        self.resize(400, 200)
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        self._create_settings_section(layout)
        self._create_buttons(layout)
    
    def _create_settings_section(self, layout):
        """Create export settings section"""
        settings_group = QGroupBox("Export Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Output type selection
        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems(['Segmentation-Matte', 'Segmentation-Alpha', 'Segmentation-BGcolor', 'Matting-Matte', 'Matting-Alpha', 'Matting-BGcolor', 'ObjectRemoval'])
        self.output_type_combo.currentTextChanged.connect(self._on_output_type_changed)
        settings_layout.addRow("Output Type:", self.output_type_combo)
        
        # Object ID selection
        self.object_id_combo = QComboBox()
        self.object_id_combo.addItem("All Objects", -1)
        if self.parent_window and hasattr(self.parent_window, 'point_manager'):
            points = self.parent_window.point_manager.get_all_points()
            if points:
                object_ids = sorted(set(point['object_id'] for point in points))
                for obj_id in object_ids:
                    self.object_id_combo.addItem(f"Object {obj_id}", obj_id)
        settings_layout.addRow("Export Object:", self.object_id_combo)
        
        # Antialiasing checkbox
        self.antialias_checkbox = QCheckBox()
        self.antialias_checkbox.setChecked(False)
        self.antialias_label = QLabel("Antialiasing:")
        settings_layout.addRow(self.antialias_label, self.antialias_checkbox)
        
        layout.addWidget(settings_group)
        
        # Initialize UI state based on default selection
        self._on_output_type_changed()
    
    def _on_output_type_changed(self):
        """Handle output type selection changes"""
        output_type = self.output_type_combo.currentText()
        
        # Enable/disable antialiasing based on whether it's a Segmentation mode
        is_segmentation = output_type.startswith('Segmentation-')
        self.antialias_checkbox.setVisible(is_segmentation)
        self.antialias_label.setVisible(is_segmentation)

        # Enable/disable object selection for ObjectRemoval mode
        is_object_removal = output_type.startswith('ObjectRemoval')
        self.object_id_combo.setEnabled(not is_object_removal)
    
    def _create_buttons(self, layout):
        """Create dialog buttons"""
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.export_btn = QPushButton("Save As...")
        self.export_btn.clicked.connect(self._save_image)
        
        self.cancel_btn = QPushButton("Close") # renamed from cancel to close
        self.cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _get_default_filename(self):
        """Generate default filename based on current settings"""
        # Get input filename if available
        input_name = "frame"
        if self.parent_window and hasattr(self.parent_window, 'settings_mgr'):
            settings_mgr = self.parent_window.settings_mgr
            input_file = settings_mgr.get_session_setting("video_file_path", "")
            if input_file:
                input_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Build filename components
        frame_num = self.frame_number
        output_type = self.output_type_combo.currentText().lower()
        
        # Add object info if specific object selected
        object_id = self.object_id_combo.currentData()
        if object_id != -1:
            object_part = f"_obj{object_id}"
        else:
            object_part = ""
        
        return f"{input_name}_frame{frame_num:04d}_{output_type}{object_part}"
    
    def _get_default_directory(self):
        """Get default directory for save dialog"""
        if self.parent_window and hasattr(self.parent_window, 'settings_mgr'):
            settings_mgr = self.parent_window.settings_mgr
            
            # Use input file directory if available
            input_file = settings_mgr.get_session_setting("video_file_path", "")
            if input_file:
                input_dir = os.path.dirname(input_file)
                if os.path.exists(input_dir):
                    return input_dir
        
        # Final fallback to current directory
        return os.getcwd()
    
    def _is_alpha_mode(self):
        """Check if current output type is an alpha mode"""
        output_type = self.output_type_combo.currentText()
        return 'Alpha' in output_type
    
    def _save_image(self):
        """Save the image using a file dialog"""
        # Generate default filename (without extension)
        default_filename = self._get_default_filename()
        default_path = os.path.join(self._get_default_directory(), default_filename)
        
        # Set up file dialog filters based on output type
        if self._is_alpha_mode():
            # Force PNG for alpha modes
            file_filter = "PNG Images (*.png);;All Files (*)"
            default_path += ".png"
        else:
            # Allow both PNG and JPEG for non-alpha modes
            file_filter = "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)"
        
        # Show save dialog
        output_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Image As",
            default_path,
            file_filter
        )
        
        if not output_path:
            return  # User cancelled
        
        # Validate file extension for alpha modes
        if self._is_alpha_mode():
            _, ext = os.path.splitext(output_path.lower())
            if ext not in ['.png']:
                # Force PNG extension for alpha modes
                if not ext:
                    output_path += '.png'
                else:
                    # Replace extension with PNG
                    output_path = os.path.splitext(output_path)[0] + '.png'
        
# Export the image
        try:
            self._export_image(output_path)
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("Export Complete")
            msg_box.setText(f"Image saved successfully to:\n{output_path}")
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
            # self.accept() # Close dialog on success
        except Exception as e:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Export Failed")
            msg_box.setText(f"Failed to save image:\n{str(e)}")
            msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            msg_box.exec()
    
    def _export_image(self, output_path):
        """Export the actual image data"""
        # Get points for mask generation
        points = self.parent_window.point_manager.get_all_points() if self.parent_window else []
        
        # Get view options based on output type
        output_type = self.output_type_combo.currentText()
        antialias = self.antialias_checkbox.isChecked()
        
        view_options = self._get_export_view_options(output_type, antialias)
        
        # Get object filter
        object_id_filter = None
        selected_object_id = self.object_id_combo.currentData()
        if selected_object_id != -1:
            object_id_filter = selected_object_id
        
        # Get image data using sammie's update_image function
        image_array = sammie.update_image(
            self.frame_number, 
            view_options, 
            points, 
            return_numpy=True, 
            object_id_filter=object_id_filter
        )
        
        if image_array is None:
            raise RuntimeError("Failed to generate image data")
        
        # Convert numpy array to PIL Image
        if len(image_array.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image_array, mode='L')
        elif image_array.shape[2] == 3:
            # RGB
            pil_image = Image.fromarray(image_array, mode='RGB')
        elif image_array.shape[2] == 4:
            # RGBA
            pil_image = Image.fromarray(image_array, mode='RGBA')
        else:
            raise RuntimeError(f"Unsupported image format: {image_array.shape}")
        
        # Determine format from file extension
        _, ext = os.path.splitext(output_path.lower())
        is_jpeg = ext in ['.jpg', '.jpeg']
        
        # Save image with appropriate settings
        save_kwargs = {}
        if is_jpeg:
            # Convert RGBA to RGB for JPEG
            if pil_image.mode == 'RGBA':
                # Create white background
                bg = Image.new('RGB', pil_image.size, (255, 255, 255))
                bg.paste(pil_image, mask=pil_image.split()[3])  # Use alpha as mask
                pil_image = bg
            
            save_kwargs['quality'] = 95
            save_kwargs['optimize'] = True
        else:  # PNG
            save_kwargs['optimize'] = True
        
        pil_image.save(output_path, **save_kwargs)
    
    def _get_export_view_options(self, output_type, antialias):
        """Get view options for export based on output type"""
        # Get bgcolor from session settings
        settings_mgr = self.parent_window.settings_mgr if self.parent_window else None
        if settings_mgr:
            bgcolor = settings_mgr.get_session_setting("bgcolor", (0, 255, 0))
        else:
            bgcolor = (0, 255, 0)

        if output_type == 'Segmentation-Matte':
            return {
                'view_mode': 'Segmentation-Matte',
                'antialias': antialias
            }
        elif output_type == 'Segmentation-Alpha':
            return {
                'view_mode': 'Segmentation-Alpha', 
                'antialias': antialias
            }
        elif output_type == 'Segmentation-BGcolor':
            return {
                'view_mode': 'Segmentation-BGcolor',
                'antialias': antialias,
                'bgcolor': bgcolor
            }
        elif output_type == 'Matting-Matte':
            return {
                'view_mode': 'Matting-Matte'
            }
        elif output_type == 'Matting-Alpha':
            return {
                'view_mode': 'Matting-Alpha'
            }
        elif output_type == 'Matting-BGcolor':
            return {
                'view_mode': 'Matting-BGcolor',
                'bgcolor': bgcolor
            }
        elif output_type == 'ObjectRemoval':
            return {
                'view_mode': 'ObjectRemoval',
                'show_removal_mask': False
            }
        else:
            # Fallback
            return {
                'view_mode': 'Segmentation-Edit',
                'show_masks': True,
                'show_outlines': False
            }