# sammie/export_dialog.py
import os
import av
import datetime
import numpy as np
import OpenEXR
import Imath
from fractions import Fraction
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QFileDialog, QProgressDialog, QApplication, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal
from sammie import sammie


class ExportWorker(QThread):
    """Worker thread for video export to prevent GUI freezing"""
    
    progress_updated = Signal(int)  # Progress percentage
    status_updated = Signal(str)    # Status message
    finished = Signal(bool, str)    # Success flag and message
    
    def __init__(self, export_params, points, total_frames, export_multiple=False, object_ids=None, parent_window=None):
        super().__init__()
        self.export_params = export_params
        self.points = points
        self.total_frames = total_frames
        self.should_cancel = False
        self.export_multiple = export_multiple
        self.object_ids = object_ids or []
        self.parent_window = parent_window
        self.object_names = export_params.get('object_names', {})
        
    def cancel(self):
        self.should_cancel = True
        
    def run(self):
        try:
            if self.export_params['codec'] == 'exr':
                self._export_exr_sequence()
            elif self.export_multiple:
                self._export_multiple_videos()
            else:
                self._export_video()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.finished.emit(False, f"Export failed: {e}\n\n{tb}")
    
    def _export_exr_sequence(self):
        """Export OpenEXR frame sequence with multiple object layers"""
        output_dir = self.export_params['output_path']
        base_filename = self.export_params['base_filename']
        include_original = self.export_params.get('include_original', False)
        
        # Get all unique object IDs from points
        all_object_ids = sorted(set(point['object_id'] for point in self.points))
        
        if not all_object_ids:
            self.finished.emit(False, "No objects found to export")
            return
        
        # Get object names from session settings
        object_names = {}
        if self.parent_window and hasattr(self.parent_window, 'settings_mgr'):
            object_names = self.parent_window.settings_mgr.get_session_setting("object_names", {})
        
        exported_files = []
        
        for frame_num in range(self.total_frames):
            if self.should_cancel:
                self.status_updated.emit("Export cancelled")
                # Clean up any partially created files
                for file_path in exported_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                self.finished.emit(False, "Export was cancelled")
                return
            
            self.status_updated.emit(f"Exporting frame {frame_num + 1}/{self.total_frames}...")
            
            # Generate filename for this frame
            frame_filename = f"{base_filename}.{frame_num:04d}.exr"
            frame_path = os.path.join(output_dir, frame_filename)
            
            try:
                # Collect all object masks for this frame
                exr_data = {}
                
                # Get matte view options
                view_options = {
                    'view_mode': self.export_params.get('output_type', 'Segmentation-Matte'),
                    'antialias': self.export_params.get('antialias', False)
                }
                
                # Export each object as a separate layer
                for obj_id in all_object_ids:
                    mask_array = sammie.update_image(
                        frame_num, view_options, self.points, 
                        return_numpy=True, object_id_filter=obj_id
                    )
                    
                    if mask_array is not None:
                        # Convert to grayscale if needed (matte should already be grayscale)
                        if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                            mask_array = mask_array[:, :, 0]  # Take first channel
                        
                        # Normalize to 0-1 range for EXR
                        if mask_array.dtype == np.uint8:
                            mask_array = mask_array.astype(np.float32) / 255.0
                        elif mask_array.dtype != np.float32:
                            mask_array = mask_array.astype(np.float32)
                        
                        # Generate layer name with object name if available
                        object_name = object_names.get(str(obj_id), "")
                        if object_name:
                            # Sanitize object name for EXR layer naming
                            sanitized_name = self._sanitize_layer_name(object_name)
                            layer_name = f'{obj_id}_{sanitized_name}'
                        else:
                            layer_name = f'{obj_id}'
                        
                        exr_data[layer_name] = mask_array
                
                # Include original frame if requested
                if include_original:
                    # Get original frame without any segmentation overlay
                    original_view_options = {
                        'view_mode': 'None',  # return original frame
                    }
                    
                    original_array = sammie.update_image(
                        frame_num, original_view_options, self.points, 
                        return_numpy=True, object_id_filter=None
                    )
                    
                    if original_array is not None:
                        # Convert to float32 and normalize if needed
                        if original_array.dtype == np.uint8:
                            original_array = original_array.astype(np.float32) / 255.0
                        elif original_array.dtype != np.float32:
                            original_array = original_array.astype(np.float32)
                        
                        # Split RGB channels for EXR
                        if len(original_array.shape) == 3 and original_array.shape[2] >= 3:
                            exr_data['original.R'] = original_array[:, :, 0]
                            exr_data['original.G'] = original_array[:, :, 1]
                            exr_data['original.B'] = original_array[:, :, 2]
                        else:
                            exr_data['original'] = original_array
                
                # Write EXR file
                if exr_data:
                    self._write_exr_file(frame_path, exr_data)
                    exported_files.append(frame_path)
                
                # Update progress
                progress = int((frame_num + 1) / self.total_frames * 100)
                self.progress_updated.emit(progress)
                
            except Exception as e:
                print (f"Error exporting frame {frame_num + 1}: {e}")
                # Clean up on error
                for file_path in exported_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                raise e
        
        if not self.should_cancel:
            self.status_updated.emit("EXR sequence export completed")
            self.finished.emit(True, f"Successfully exported {len(exported_files)} EXR frames to {output_dir}")

    def _sanitize_layer_name(self, name):
        """Sanitize object name for use in EXR layer naming"""
        import re
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length to reasonable size
        if len(sanitized) > 20:
            sanitized = sanitized[:20]
        return sanitized if sanitized else "unnamed"

    def _sanitize_filename_component(self, name):
        """Sanitize name for use in filenames"""
        import re
        if not name:
            return ""
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length to reasonable size
        if len(sanitized) > 20:
            sanitized = sanitized[:20]
        return sanitized if sanitized else "unnamed"

    def _write_exr_file(self, filepath, data_dict):
        """Write EXR file with multiple layers using Python OpenEXR"""
        try:
            first_layer = next(iter(data_dict.values()))
            height, width = first_layer.shape

            header = OpenEXR.Header(width, height)
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

            # Declare channels
            for name in data_dict.keys():
                header['channels'][name] = Imath.Channel(FLOAT)

            # Enable PIZ compression
            header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)

            out = OpenEXR.OutputFile(filepath, header)

            # Prepare channel data
            channels = {}
            for name, arr in data_dict.items():
                if arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                arr = np.ascontiguousarray(arr)
                channels[name] = arr.tobytes()

            out.writePixels(channels)
            out.close()

        except Exception as e:
            raise RuntimeError(f"Failed to write EXR file: {e}")
    
    def _export_multiple_videos(self):
        """Export individual videos for each object ID"""
        total_exports = len(self.object_ids)
        total_progress_steps = total_exports * self.total_frames
        current_step = 0
        
        exported_files = []
        
        for i, object_id in enumerate(self.object_ids):
            if self.should_cancel:
                self.status_updated.emit("Export cancelled")
                # Clean up any partially created files
                for file_path in exported_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                self.finished.emit(False, "Export was cancelled")
                return
            
            # Update export params for current object
            current_params = self.export_params.copy()
            current_params['object_id'] = object_id
            
            # Generate output path for this object
            output_path = self._generate_object_output_path(object_id)
            current_params['output_path'] = output_path
            
            self.status_updated.emit(f"Exporting object {object_id} ({i+1}/{total_exports})...")
            
            try:
                self._export_single_video(current_params, current_step, total_progress_steps)
                exported_files.append(output_path)
                current_step += self.total_frames
            except Exception as e:
                # Clean up any partially created files
                for file_path in exported_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                raise e
        
        if not self.should_cancel:
            self.status_updated.emit("All exports completed successfully")
            files_text = "\n".join([os.path.basename(f) for f in exported_files])
            self.finished.emit(True, f"Successfully exported {len(exported_files)} videos:\n{files_text}")
    
    def _generate_object_output_path(self, object_id):
        """Generate output path for a specific object ID using the filename template"""
        # Get the template and directory from export params
        template = self.export_params['filename_template']
        directory = self.export_params['output_path']  # This is now just the folder path
        
        # Resolve the template with the specific object ID
        resolved_name = self._resolve_template_for_object(template, object_id)
        
        # Get extension from codec
        codec = self.export_params['codec']
        if codec == 'prores':
            ext = '.mov'
        elif codec == 'ffv1':
            ext = '.mkv'
        elif codec == 'vp9':
            ext = '.webm'
        else:
            ext = '.mp4'
        
        return os.path.join(directory, resolved_name + ext)
    
    def _resolve_template_for_object(self, template, object_id):
        """Resolve filename template for a specific object ID"""
        base_params = self.export_params
        now = datetime.datetime.now()
        
        # Get object names from parent window if available
        object_name = ""
        if self.parent_window and hasattr(self.parent_window, 'settings_mgr'):
            object_names = self.parent_window.settings_mgr.get_session_setting("object_names", {})
            object_name = object_names.get(str(object_id), "")
        elif hasattr(self, 'object_names'):
            object_name = self.object_names.get(str(object_id), "")

        # Provide default name if empty
        if not object_name.strip():
            if object_id == -1:
                object_name = "all"
            else:
                object_name = f"object_{object_id}"

        sanitized_object_name = self._sanitize_filename_component(object_name)

        tag_values = {
            "input_name": base_params.get('input_name', 'video'),
            "output_type": base_params.get('output_type', ''),
            "codec": base_params.get('codec', ''),
            "object_id": str(object_id),
            "object_name": sanitized_object_name,
            "date": now.strftime("%Y%m%d"),
            "time": now.strftime("%H%M%S"),
            "datetime": now.strftime("%Y%m%d_%H%M%S"),
        }
        
        result = template
        for tag, value in tag_values.items():
            result = result.replace(f"{{{tag}}}", value)
        return result
    
    def _export_video(self):
        """Main export logic for single video"""
        self._export_single_video(self.export_params, 0, self.total_frames)
        
        if not self.should_cancel:
            self.status_updated.emit("Export completed successfully")
            self.finished.emit(True, f"Video exported successfully to {self.export_params['output_path']}")
    
    def _export_single_video(self, params, progress_offset, total_progress_steps):
        """Export a single video with given parameters"""
        output_path = params['output_path']
        codec = params['codec']
        output_type = params['output_type']
        antialias = params['antialias']
        quantizer = params.get('quantizer', 14)
        object_id_filter = None if params['object_id'] == -1 else params['object_id']
        
        # Create output container and stream
        container = av.open(output_path, mode='w')
        
        # Convert float framerates to fraction
        fps = sammie.VideoInfo.fps
        if fps == 29.97:
            fps_rational = Fraction(30000, 1001)
        elif fps == 23.976:
            fps_rational = Fraction(24000, 1001)
        elif fps == 59.94:
            fps_rational = Fraction(60000, 1001)
        else:
            fps_rational = Fraction(fps)
            fps_rational = fps_rational.limit_denominator(0x7fffffff)
        
        # Configure codec options
        codec_options = {}
        if codec == 'prores':
            stream = container.add_stream('prores_ks', rate=fps_rational)
            stream.pix_fmt = 'yuv422p10le'
            codec_options = {'profile': '3'}  # ProRes HQ
        elif codec == 'ffv1':
            stream = container.add_stream('ffv1', rate=fps_rational)
            codec_options = {'level': '3'}
            stream.pix_fmt = 'bgr0'
        elif codec == 'vp9':
            stream = container.add_stream('libvpx-vp9', rate=fps_rational)
            codec_options = {
                'crf': str(quantizer),
                'b': '0',  # Use constant quality mode
                'row-mt': '1'  # Enable row-based multithreading
            }
            stream.pix_fmt = 'yuv420p'
        elif codec == 'x264':
            stream = container.add_stream('libx264', rate=fps_rational)
            codec_options = {'crf': str(quantizer), 'preset': 'medium'}
            stream.pix_fmt = 'yuv420p'
        elif codec == 'x265':
            stream = container.add_stream('libx265', rate=fps_rational)
            codec_options = {'crf': str(quantizer), 'preset': 'medium'}
            stream.pix_fmt = 'yuv420p'
        
        # Set stream properties
        stream.width = sammie.VideoInfo.width
        stream.height = sammie.VideoInfo.height
        
        # Set custom options for alpha channel
        has_alpha = ('Alpha' in output_type)
        if has_alpha:
            if codec == 'prores':
                stream.pix_fmt='yuva444p10le'
                codec_options = {'profile': '4'}  # 4444
            elif codec == 'ffv1':
                stream.pix_fmt = 'bgra'
            elif codec == 'vp9':
                stream.pix_fmt = 'yuva420p'
        
        # Set codec options
        for key, value in codec_options.items():
            stream.options[key] = value
        
        try:
            # Process each frame
            for frame_num in range(self.total_frames):
                if self.should_cancel:
                    self.status_updated.emit("Export cancelled")
                    container.close()
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    self.finished.emit(False, "Export was cancelled")
                    return
                
                # Get the appropriate view options for export
                view_options = self._get_export_view_options(output_type, antialias)
                
                # Get frame data using existing update_image function with numpy return
                frame_array = sammie.update_image(frame_num, view_options, self.points, return_numpy=True, object_id_filter=object_id_filter)
                if frame_array is None:
                    continue
                
                # Ensure frame_array is in the correct format for AV
                if has_alpha and frame_array.shape[2] != 4:
                    # Add alpha channel if missing
                    alpha_channel = np.full((frame_array.shape[0], frame_array.shape[1], 1), 255, dtype=np.uint8)
                    frame_array = np.concatenate([frame_array, alpha_channel], axis=2)
                elif not has_alpha and frame_array.shape[2] == 4:
                    # Remove alpha channel if not needed
                    frame_array = frame_array[:, :, :3]
                
                # Create AV frame - handle color channel ordering for AV
                if has_alpha:
                    # RGBA format for alpha
                    av_frame = av.VideoFrame.from_ndarray(frame_array, format='rgba')
                else:
                    # RGB format for standard video
                    av_frame = av.VideoFrame.from_ndarray(frame_array, format='rgb24')
                    
                av_frame.pts = frame_num
                
                # Encode and write frame
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                
                # Update progress
                current_progress_step = progress_offset + frame_num + 1
                progress = int(current_progress_step / total_progress_steps * 100)
                self.progress_updated.emit(progress)
            
            # Flush remaining frames
            for packet in stream.encode():
                container.mux(packet)
                
        finally:
            container.close()
    
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
            return {
                'view_mode': 'Segmentation-Edit',
                'show_masks': True,
                'show_outlines': False
            }


class ExportDialog(QDialog):
    """Dialog for configuring video export settings"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.export_worker = None
        self.progress_dialog = None
        self.setWindowTitle("Export Video")
        self.setModal(True)
        self.resize(500, 520)
        self._init_ui()

        
    def _init_ui(self):
        """Initialize the dialog UI"""
        layout = QVBoxLayout(self)
        self._create_output_section(layout)
        self._create_settings_section(layout)
        self._create_buttons(layout)
        self._update_filename_preview()
        self._load_saved_settings()
    
    def _create_output_section(self, layout):
        """Create output file selection section"""
        output_group = QGroupBox("Output File")
        output_layout = QFormLayout(output_group)
        
        # Output folder selection
        folder_layout = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select output folder...")
        folder_layout.addWidget(self.folder_edit)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_output_folder)
        folder_layout.addWidget(self.browse_btn)

        output_layout.addRow("Output Folder:", folder_layout)

        # Checkbox to use input folder
        self.use_input_folder_checkbox = QCheckBox("Use same folder as input file")
        self.use_input_folder_checkbox.stateChanged.connect(self._on_use_input_folder_changed)
        output_layout.addRow("", self.use_input_folder_checkbox)
        
        # Filename template with tag dropdown
        template_layout = QHBoxLayout()
        self.filename_template_edit = QLineEdit()
        self.filename_template_edit.setPlaceholderText("{input_name}-{output_type}")
        self.filename_template_edit.textChanged.connect(self._update_filename_preview)
        template_layout.addWidget(self.filename_template_edit)

        self.tag_dropdown = QComboBox()
        self.tag_dropdown.addItem("Insert tag...")  # placeholder
        self.tag_dropdown.addItems([
        "{input_name}", "{output_type}", "{object_id}", 
        "{object_name}", "{codec}", "{date}", "{time}", 
        "{datetime}"
        ])
        self.tag_dropdown.currentIndexChanged.connect(self._insert_tag_into_template)
        template_layout.addWidget(self.tag_dropdown)

        output_layout.addRow("Filename:", template_layout)
        
        # Preview label
        self.filename_preview_label = QLabel()
        self.filename_preview_label.setStyleSheet("color: gray; font-style: italic;")
        self.filename_preview_label.setWordWrap(True)
        output_layout.addRow("Preview:", self.filename_preview_label)
        
        # Codec selection
        self.codec_combo = QComboBox()
        self.codec_combo.addItems(['prores', 'ffv1', 'x264', 'x265', 'vp9', 'exr'])
        self.codec_combo.currentTextChanged.connect(self._on_codec_changed)
        output_layout.addRow("Codec:", self.codec_combo)

        # Alpha unsupported message (hidden by default)
        self.alpha_warning_label = QLabel("Alpha channel is not supported with x264 or x265.")
        self.alpha_warning_label.setStyleSheet("color: orange;")
        self.alpha_warning_label.setWordWrap(True)
        self.alpha_warning_label.setVisible(False)
        output_layout.addRow("", self.alpha_warning_label)
        
        layout.addWidget(output_group)
    
    def _insert_tag_into_template(self, index):
        if index <= 0:
            return  # Ignore placeholder
        tag = self.tag_dropdown.itemText(index)
        cursor_pos = self.filename_template_edit.cursorPosition()
        text = self.filename_template_edit.text()
        new_text = text[:cursor_pos] + tag + text[cursor_pos:]
        self.filename_template_edit.setText(new_text)
        # Reset dropdown back to placeholder
        self.tag_dropdown.setCurrentIndex(0)
        # Move cursor after inserted tag
        self.filename_template_edit.setCursorPosition(cursor_pos + len(tag))

    def _create_settings_section(self, layout):
        """Create export settings section"""
        settings_group = QGroupBox("Export Settings")
        settings_layout = QFormLayout(settings_group)
        
        # Output type selection
        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems([
            'Segmentation-Matte', 'Segmentation-Alpha', 'Segmentation-BGcolor',
            'Matting-Matte', 'Matting-Alpha', 'Matting-BGcolor', 'ObjectRemoval'
        ])
        self.output_type_combo.currentTextChanged.connect(self._on_output_type_changed)
        settings_layout.addRow("Output Type:", self.output_type_combo)
        
        # Object ID selection
        self.object_id_combo = QComboBox()
        self.object_id_combo.addItem("All Objects", -1)  # -1 means all objects
        if self.parent_window and hasattr(self.parent_window, 'point_manager'):
            points = self.parent_window.point_manager.get_all_points()
            if points:
                object_ids = sorted(set(point['object_id'] for point in points))
                for obj_id in object_ids:
                    self.object_id_combo.addItem(f"Object {obj_id}", obj_id)
        self.object_id_combo.currentIndexChanged.connect(self._update_filename_preview)
        settings_layout.addRow("Export Object:", self.object_id_combo)

        # Export multiple objects checkbox
        self.export_multiple_checkbox = QCheckBox("Export videos for each object")
        self.export_multiple_checkbox.stateChanged.connect(self._on_export_multiple_changed)
        settings_layout.addRow("", self.export_multiple_checkbox)

        # Include original frame checkbox (for EXR only)
        self.include_original_checkbox = QCheckBox("Include original frame as layer")
        self.include_original_checkbox.setVisible(False)
        settings_layout.addRow("", self.include_original_checkbox)
        
        # Quantizer setting (for x264/x265)
        self.quantizer_spin = QSpinBox()
        self.quantizer_spin.setRange(0, 51)
        self.quantizer_spin.setValue(14)
        self.quantizer_spin.setToolTip("Lower values = higher quality, larger file size")
        settings_layout.addRow("Quality (CRF):", self.quantizer_spin)
        self.quantizer_label = settings_layout.labelForField(self.quantizer_spin)
        
        # Antialiasing checkbox
        self.antialias_checkbox = QCheckBox()
        self.antialias_checkbox.setChecked(False)
        self.antialias_label = QLabel("Antialiasing:")
        settings_layout.addRow(self.antialias_label, self.antialias_checkbox)
        
        layout.addWidget(settings_group)
        
        # Initialize UI state based on default selection
        self._on_codec_changed(self.codec_combo.currentText())
        self._on_output_type_changed()

    def _on_output_type_changed(self):
        """Handle output type selection changes"""
        output_type = self.output_type_combo.currentText()
        
        # Show/hide antialiasing based on whether it's a Segmentation mode
        is_segmentation = output_type.startswith('Segmentation-')
        self.antialias_checkbox.setVisible(is_segmentation)
        self.antialias_label.setVisible(is_segmentation)

        # Show/hide object selection for ObjectRemoval mode
        is_object_removal = output_type.startswith('ObjectRemoval')
        self.object_id_combo.setEnabled(not is_object_removal)
        self.export_multiple_checkbox.setVisible(not is_object_removal)
        
        # Update filename preview
        self._update_filename_preview()

    def _on_export_multiple_changed(self, state):
        """Handle changes to the export multiple checkbox"""
        export_multiple = self.export_multiple_checkbox.isChecked()
        
        # Disable/enable object selection
        self.object_id_combo.setEnabled(not export_multiple)
        
        # Update filename preview
        self._update_filename_preview()
        
    def _create_buttons(self, layout):
        """Create dialog buttons"""
        button_layout = QHBoxLayout()
        
        self.save_settings_btn = QPushButton("Save Settings")
        self.save_settings_btn.clicked.connect(self._save_current_settings)
        button_layout.addWidget(self.save_settings_btn)
        
        button_layout.addStretch()
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._start_export)
        
        self.cancel_btn = QPushButton("Close") # renamed from cancel to close
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
    
# === Folder and preview helpers ===
    def _browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder:
            self.folder_edit.setText(folder)
            self._update_filename_preview()

    def _on_use_input_folder_changed(self, state):
        use_input = self.use_input_folder_checkbox.isChecked()
        self.folder_edit.setEnabled(not use_input)
        self.browse_btn.setEnabled(not use_input)
        self._update_filename_preview()

    def _get_available_object_ids(self):
        """Get list of available object IDs from points"""
        if self.parent_window and hasattr(self.parent_window, 'point_manager'):
            points = self.parent_window.point_manager.get_all_points()
            if points:
                return sorted(set(point['object_id'] for point in points))
        return []
    
    def _resolve_filename_template(self, template, object_id=None):
        """Replace tags with actual values for preview/usage"""
        settings_mgr = self.parent_window.settings_mgr
        input_file = settings_mgr.get_session_setting("video_file_path", "")
        base_name = os.path.splitext(os.path.basename(input_file))[0] if input_file else "video"
        input_ext = os.path.splitext(input_file)[1] if input_file else ""
        input_path = os.path.dirname(input_file) if input_file else ""
        
        now = datetime.datetime.now()

        # Use provided object_id or get from combo box
        if object_id is not None:
            selected_object_id = object_id
        else:
            selected_object_id = self.object_id_combo.currentData() if self.object_id_combo else -1

        # Get object name from session settings and sanitize it
        object_names = settings_mgr.get_session_setting("object_names", {})
        if selected_object_id != -1:
            raw_object_name = object_names.get(str(selected_object_id), "")
            # Provide default name if empty
            if not raw_object_name.strip():
                raw_object_name = f"object_{selected_object_id}"
            sanitized_object_name = self._sanitize_filename_component(raw_object_name)
        else:
            sanitized_object_name = "all"

        tag_values = {
            "input_name": base_name,
            "input_ext": input_ext,
            "input_path": input_path,
            "output_type": self.output_type_combo.currentText() if self.output_type_combo else "",
            "codec": self.codec_combo.currentText() if self.codec_combo else "",
            "object_id": str(selected_object_id) if selected_object_id != -1 else "all",
            "object_name": sanitized_object_name,
            "date": now.strftime("%Y%m%d"),
            "time": now.strftime("%H%M%S"),
            "datetime": now.strftime("%Y%m%d_%H%M%S"),
        }

        result = template
        for tag, value in tag_values.items():
            result = result.replace(f"{{{tag}}}", value)
        return result
    
    def _update_filename_preview(self):
        """Update filename preview based on current settings"""
        template = self.filename_template_edit.text().strip() or "{input_name}-{output_type}"

        # Extension from codec
        codec = self.codec_combo.currentText()
        if codec == 'prores':
            ext = '.mov'
        elif codec == 'ffv1':
            ext = '.mkv'
        elif codec == 'vp9':
            ext = '.webm'
        elif codec == 'exr':
            ext = '.####.exr'  # Frame sequence pattern
        else:
            ext = '.mp4'
        
        if self.use_input_folder_checkbox.isChecked():
            settings_mgr = self.parent_window.settings_mgr
            input_file = settings_mgr.get_session_setting("video_file_path", "")
            folder = os.path.dirname(input_file) if input_file else os.getcwd()
        else:
            folder = self.folder_edit.text() or os.getcwd()
        
        # Special handling for EXR sequences
        if codec == 'exr':
            resolved_name = self._resolve_filename_template(template)
            preview_text = f"{folder}/{resolved_name}{ext}"
            self.filename_preview_label.setText(preview_text)
            return os.path.join(folder, resolved_name)  # Return base name without extension for EXR
        
        if self.export_multiple_checkbox.isChecked():
            # Show preview for multiple files
            object_ids = self._get_available_object_ids()
            if object_ids:
                # Show preview for first few objects
                preview_files = []
                for i, obj_id in enumerate(object_ids[:3]):  # Show first 3 as examples
                    resolved_name = self._resolve_filename_template(template, obj_id)
                    full_path = os.path.join(folder, resolved_name + ext)
                    preview_files.append(os.path.basename(full_path))
                
                preview_text = "\n".join(preview_files)
                if len(object_ids) > 3:
                    preview_text += f"\n... and {len(object_ids) - 3} more files"
                
                self.filename_preview_label.setText(preview_text)
                return os.path.join(folder, self._resolve_filename_template(template, object_ids[0]) + ext)
            else:
                self.filename_preview_label.setText("No objects found")
                return ""
        else:
            # Single file preview
            resolved_name = self._resolve_filename_template(template)
            full_path = os.path.join(folder, resolved_name + ext)
            self.filename_preview_label.setText(full_path)
            return full_path
    
    def _on_codec_changed(self, codec):
        """Handle codec selection changes"""
        is_x264_x265 = codec in ['x264', 'x265']
        is_vp9 = codec == 'vp9'
        is_exr = codec == 'exr'
        
        # Show/hide quantizer setting based on codec
        self.quantizer_spin.setVisible(is_x264_x265 or is_vp9)
        self.quantizer_label.setVisible(is_x264_x265 or is_vp9)
        
        # Adjust quantizer range for VP9 (0-63 instead of 0-51)
        if is_vp9:
            self.quantizer_spin.setRange(0, 63)
            self.quantizer_spin.setValue(14)  # Default CQ level for VP9
            self.quantizer_label.setText("Quality (CQ):")
        elif is_x264_x265:
            self.quantizer_spin.setRange(0, 51)
            self.quantizer_spin.setValue(14)
            self.quantizer_label.setText("Quality (CRF):")

        # Handle Alpha option availability
        current_output_type = self.output_type_combo.currentText()
        
        # Store current alpha types
        alpha_types = ['Segmentation-Alpha', 'Matting-Alpha']
        
        if (is_x264_x265 or is_exr):
            # Remove alpha options if present
            for alpha_type in alpha_types:
                alpha_index = self.output_type_combo.findText(alpha_type)
                if alpha_index >= 0:
                    # If an Alpha type is currently selected, switch to corresponding Matte first
                    if current_output_type == alpha_type:
                        if alpha_type == 'Segmentation-Alpha':
                            matte_index = self.output_type_combo.findText('Segmentation-Matte')
                        else:  # Matting-Alpha
                            matte_index = self.output_type_combo.findText('Matting-Matte')
                        
                        if matte_index >= 0:
                            self.output_type_combo.setCurrentIndex(matte_index)
                    
                    # Remove the alpha option
                    self.output_type_combo.removeItem(alpha_index)
        else:
            # Re-add alpha options if they're missing (maintaining proper order)
            all_types = [
                'Segmentation-Matte', 'Segmentation-Alpha', 'Segmentation-BGcolor',
                'Matting-Matte', 'Matting-Alpha', 'Matting-BGcolor', 'ObjectRemoval'
            ]
            
            # Check which alpha types are missing and add them in correct positions
            for i, type_name in enumerate(all_types):
                if type_name in alpha_types and self.output_type_combo.findText(type_name) == -1:
                    self.output_type_combo.insertItem(i, type_name)

        # Show/hide warning label
        self.alpha_warning_label.setVisible(is_x264_x265)

        # EXR-specific UI changes
        if is_exr:
            # For EXR, only allow Segmentation-Matte and Matting-Matte options
            allowed_types = ['Segmentation-Matte', 'Matting-Matte']
            
            # Save current selection if it's allowed
            current_selection = self.output_type_combo.currentText()
            
            # Clear and rebuild combo with only allowed options
            self.output_type_combo.clear()
            self.output_type_combo.addItems(allowed_types)
            
            # Restore selection if it was allowed, otherwise default to Segmentation-Matte
            if current_selection in allowed_types:
                index = self.output_type_combo.findText(current_selection)
                if index >= 0:
                    self.output_type_combo.setCurrentIndex(index)
            else:
                # Default to Segmentation-Matte
                self.output_type_combo.setCurrentIndex(0)
            
            # Keep output type combo enabled for EXR
            self.output_type_combo.setEnabled(True)
            
            # Force All Objects and disable object selection
            self.object_id_combo.setCurrentIndex(0)  # All Objects
            self.object_id_combo.setEnabled(False)
            
            # Hide export multiple checkbox
            self.export_multiple_checkbox.setVisible(False)
            self.export_multiple_checkbox.setChecked(False)
            
            # Show include original checkbox
            self.include_original_checkbox.setVisible(True)
            
            # Update label text to indicate frame sequence
            if hasattr(self, 'filename_preview_label'):
                current_text = self.filename_preview_label.text()
                if current_text and not "Frame sequence:" in current_text:
                    self.filename_preview_label.setText(f"Frame sequence: {current_text}")
        else:
            # Re-enable controls for non-EXR codecs and restore full options
            all_types = [
                'Segmentation-Matte', 'Segmentation-Alpha', 'Segmentation-BGcolor',
                'Matting-Matte', 'Matting-Alpha', 'Matting-BGcolor', 'ObjectRemoval'
            ]
            
            # Save current selection
            current_selection = self.output_type_combo.currentText()
            
            # Rebuild combo with all options (excluding alpha if x264/x265)
            self.output_type_combo.clear()
            for type_name in all_types:
                if not (is_x264_x265 and 'Alpha' in type_name):
                    self.output_type_combo.addItem(type_name)
            
            # Restore selection or use default
            index = self.output_type_combo.findText(current_selection)
            if index >= 0:
                self.output_type_combo.setCurrentIndex(index)
            
            self.output_type_combo.setEnabled(True)
            self.object_id_combo.setEnabled(not self.export_multiple_checkbox.isChecked())
            self.export_multiple_checkbox.setVisible(True)
            self.include_original_checkbox.setVisible(False)
            
        self._update_filename_preview()
    
    def _sanitize_filename_component(self, name):
        """Sanitize name for use in filenames"""
        import re
        if not name:
            return ""
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w]', '_', name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Limit length to reasonable size
        if len(sanitized) > 20:
            sanitized = sanitized[:20]
        return sanitized if sanitized else "unnamed"
    
    def _validate_settings(self):
        """Validate export settings"""
        # Check output directory
        if self.use_input_folder_checkbox.isChecked():
            if self.parent_window and hasattr(self.parent_window, 'settings_mgr'):
                settings_mgr = self.parent_window.settings_mgr
                input_file = settings_mgr.get_session_setting("video_file_path", "")
                folder = os.path.dirname(input_file) if input_file else ""
                if not folder or not os.path.exists(folder):
                    QMessageBox.warning(self, "Invalid Settings", 
                                      "Input file directory is not available. Please select an output folder manually.")
                    return False
            else:
                QMessageBox.warning(self, "Invalid Settings", 
                                  "Cannot determine input file location. Please select an output folder manually.")
                return False
        else:
            folder = self.folder_edit.text().strip()
            if not folder:
                QMessageBox.warning(self, "Invalid Settings", "Please select an output folder.")
                return False
            # Prompt to create folder if it doesn't exist
            if not os.path.exists(folder):
                reply = QMessageBox.question(
                    self, "Create Folder?",
                    f"The output folder does not exist:\n\n{folder}\n\nDo you want to create it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply != QMessageBox.Yes:
                    return False
                
                try:
                    os.makedirs(folder, exist_ok=True)
                except OSError as e:
                    QMessageBox.warning(self, "Invalid Settings", 
                                    f"Could not create output directory:\n{e}")
                    return False
        
        template = self.filename_template_edit.text().strip() or "{input_name}-{output_type}"
    
        if "{object_name}" in template:
            # Check if we have object names available
            object_names = {}
            if hasattr(self.parent_window, 'settings_mgr'):
                object_names = self.parent_window.settings_mgr.get_session_setting("object_names", {})
            
            # Get object IDs that will be exported
            if self.export_multiple_checkbox.isChecked():
                object_ids = self._get_available_object_ids()
            else:
                selected_object_data = self.object_id_combo.currentData()
                object_ids = [selected_object_data] if selected_object_data != -1 else self._get_available_object_ids()
        
        # EXR-specific validation
        codec = self.codec_combo.currentText()
        if codec == 'exr':
            
            # Check for objects
            object_ids = self._get_available_object_ids()
            if not object_ids:
                QMessageBox.warning(self, "No Objects", "No objects found to export.")
                return False
            
            # Check for existing files
            template = self.filename_template_edit.text().strip() or "{input_name}-{output_type}"
            resolved_name = self._resolve_filename_template(template)
            
            if self.use_input_folder_checkbox.isChecked():
                settings_mgr = self.parent_window.settings_mgr
                input_file = settings_mgr.get_session_setting("video_file_path", "")
                folder = os.path.dirname(input_file) if input_file else os.getcwd()
            else:
                folder = self.folder_edit.text() or os.getcwd()
            
            # Check if any EXR files with this pattern exist
            existing_files = []
            total_frames = sammie.VideoInfo.total_frames
            for frame_num in range(min(5, total_frames)):  # Check first 5 frames
                frame_filename = f"{resolved_name}.{frame_num:04d}.exr"
                frame_path = os.path.join(folder, frame_filename)
                if os.path.exists(frame_path):
                    existing_files.append(frame_filename)
            
            if existing_files:
                files_text = "\n".join(existing_files)
                if len(existing_files) == 5 and total_frames > 5:
                    files_text += f"\n... (and possibly {total_frames - 5} more)"
                
                reply = QMessageBox.question(
                    self, "EXR Files Exist",
                    f"EXR files with this naming pattern already exist:\n\n{files_text}\n\nDo you want to overwrite them?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return False
            
            return True
        
        # Special validation for multiple export mode
        if self.export_multiple_checkbox.isChecked():
            template = self.filename_template_edit.text().strip() or "{input_name}-{output_type}"
            
            # Check if either {object_id} or {object_name} tag is included
            has_object_id = "{object_id}" in template
            has_object_name = "{object_name}" in template
            
            if not has_object_id and not has_object_name:
                QMessageBox.warning(
                    self, "Invalid Settings", 
                    "When exporting videos for each object, the filename must include either {object_id} or {object_name} tag to avoid overwriting files."
                )
                return False
            
            # Check for existing files and confirm overwrite
            object_ids = self._get_available_object_ids()
            if not object_ids:
                QMessageBox.warning(self, "No Objects", "No objects found to export.")
                return False
            
            existing_files = []
            for obj_id in object_ids:
                output_path = self._generate_output_path_for_object(obj_id)
                if os.path.exists(output_path):
                    existing_files.append(os.path.basename(output_path))
            
            if existing_files:
                files_text = "\n".join(existing_files[:5])  # Show first 5
                if len(existing_files) > 5:
                    files_text += f"\n... and {len(existing_files) - 5} more files"
                
                reply = QMessageBox.question(
                    self, "Files Exist",
                    f"The following files already exist and will be overwritten:\n\n{files_text}\n\nDo you want to continue?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return False
        else:
            # Single file validation (existing logic)
            output_path = self._update_filename_preview()
            if os.path.exists(output_path):
                reply = QMessageBox.question(
                    self, "File Exists",
                    f"The file '{os.path.basename(output_path)}' already exists.\n\nDo you want to overwrite it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return False

        return True

    def _generate_output_path_for_object(self, object_id):
        """Generate full output path for a specific object ID"""
        template = self.filename_template_edit.text().strip() or "{input_name}-{output_type}"
        resolved_name = self._resolve_filename_template(template, object_id)
        
        # Extension from codec
        codec = self.codec_combo.currentText()
        if codec == 'prores':
            ext = '.mov'
        elif codec == 'ffv1':
            ext = '.mkv'
        else:
            ext = '.mp4'
        
        if self.use_input_folder_checkbox.isChecked():
            settings_mgr = self.parent_window.settings_mgr
            input_file = settings_mgr.get_session_setting("video_file_path", "")
            folder = os.path.dirname(input_file) if input_file else os.getcwd()
        else:
            folder = self.folder_edit.text() or os.getcwd()
        
        return os.path.join(folder, resolved_name + ext)
    
    def _start_export(self):
        """Start the export process"""
        if not self._validate_settings():
            return
        
        # Check if tracking or matting has been completed
        codec = self.codec_combo.currentText()
        output_type = self.output_type_combo.currentText()
        if output_type.startswith('Segmentation'):
            if not self.parent_window.sam_manager.propagated:
                reply = QMessageBox.question(
                    self, "No Tracking Data",
                    "Objects have not been tracked yet. Only frames with manually added points will have masks. Continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
        elif output_type.startswith('Matting'):
            if not self.parent_window.matany_manager.propagated:
                reply = QMessageBox.question(
                    self, "No Matting Data",
                    "Matting data may not exist or may be outdated. Continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
        elif output_type.startswith('ObjectRemoval'):
            if not self.parent_window.sam_manager.propagated:
                reply = QMessageBox.question(
                    self, "No Tracking Data",
                    "Objects have not been tracked yet. Object removal data may not exist or may be outdated. Continue?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
        
        # Get points and total frames
        points = self.parent_window.point_manager.get_all_points()
        total_frames = sammie.VideoInfo.total_frames
        
        if total_frames == 0:
            QMessageBox.warning(self, "Export Error", "No video data available for export.")
            return
        
        # Determine export mode and parameters
        export_multiple = self.export_multiple_checkbox.isChecked()
        
        if codec == 'exr':
            # EXR export mode
            template = self.filename_template_edit.text().strip() or "{input_name}-{output_type}"
            resolved_name = self._resolve_filename_template(template)
            
            if self.use_input_folder_checkbox.isChecked():
                settings_mgr = self.parent_window.settings_mgr
                input_file = settings_mgr.get_session_setting("video_file_path", "")
                folder = os.path.dirname(input_file) if input_file else os.getcwd()
            else:
                folder = self.folder_edit.text() or os.getcwd()
            
            export_params = {
                'output_path': folder,
                'base_filename': resolved_name,
                'codec': 'exr',
                'output_type': self.output_type_combo.currentText(),
                'antialias': self.antialias_checkbox.isChecked(),
                'include_original': self.include_original_checkbox.isChecked(),
                'object_id': -1  # Always all objects for EXR
            }
            
            # Create worker for EXR export
            self.export_worker = ExportWorker(export_params, points, total_frames, parent_window=self.parent_window)
            
        elif export_multiple:
            # Multiple object export
            object_ids = self._get_available_object_ids()
            if not object_ids:
                QMessageBox.warning(self, "No Objects", "No objects found to export.")
                return
            
            # Get folder path
            if self.use_input_folder_checkbox.isChecked():
                settings_mgr = self.parent_window.settings_mgr
                input_file = settings_mgr.get_session_setting("video_file_path", "")
                folder = os.path.dirname(input_file) if input_file else os.getcwd()
                input_name = os.path.splitext(os.path.basename(input_file))[0] if input_file else "video"
            else:
                folder = self.folder_edit.text() or os.getcwd()
                settings_mgr = self.parent_window.settings_mgr
                input_file = settings_mgr.get_session_setting("video_file_path", "")
                input_name = os.path.splitext(os.path.basename(input_file))[0] if input_file else "video"
            
            # Get filename template
            template = self.filename_template_edit.text().strip() or "{input_name}-{output_type}"
            
            object_names = {}
            if hasattr(self.parent_window, 'settings_mgr'):
                object_names = self.parent_window.settings_mgr.get_session_setting("object_names", {})
            
            export_params = {
                'output_path': folder,  # Just the directory for multiple export
                'filename_template': template,
                'codec': self.codec_combo.currentText(),
                'output_type': self.output_type_combo.currentText(),
                'antialias': self.antialias_checkbox.isChecked(),
                'quantizer': self.quantizer_spin.value(),
                'object_id': -1,  # Will be set per object by worker
                'object_names': object_names,
                'input_name': input_name
            }
            
            # Create worker for multiple export
            self.export_worker = ExportWorker(export_params, points, total_frames, export_multiple=True, 
                                              object_ids=object_ids, parent_window=self.parent_window)
        else:
            # Single export (existing logic)
            selected_object_data = self.object_id_combo.currentData()
            output_path = self._update_filename_preview()
            export_params = {
                'output_path': output_path,
                'codec': self.codec_combo.currentText(),
                'output_type': self.output_type_combo.currentText(),
                'antialias': self.antialias_checkbox.isChecked(),
                'quantizer': self.quantizer_spin.value(),
                'object_id': selected_object_data  # -1 for all objects, specific ID for single object
            }
            
            # Create worker for single export
            self.export_worker = ExportWorker(export_params, points, total_frames, parent_window=self.parent_window)
        
        # Connect worker signals
        self.export_worker.progress_updated.connect(self._update_progress)
        self.export_worker.status_updated.connect(self._update_status)
        self.export_worker.finished.connect(self._export_finished)
        
        # Create progress dialog
        if codec == 'exr':
            initial_text = f"Exporting {total_frames} EXR frames..."
        elif export_multiple:
            object_ids = self._get_available_object_ids()
            initial_text = f"Exporting {len(object_ids)} videos..."
        else:
            initial_text = "Exporting video..."
            
        self.progress_dialog = QProgressDialog(initial_text, "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Exporting Video" if codec != 'exr' else "Exporting EXR Sequence")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.canceled.connect(self._cancel_export)
        self.progress_dialog.show()
        
        # Disable dialog controls during export
        self.export_btn.setEnabled(False)
        
        # Start export
        self.export_worker.start()
    
    def _update_progress(self, value):
        """Update progress dialog"""
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
    
    def _update_status(self, message):
        """Update status message"""
        if self.progress_dialog:
            self.progress_dialog.setLabelText(message)
    
    def _cancel_export(self):
        """Cancel the export process"""
        if self.export_worker:
            self.export_worker.cancel()
    
    def _export_finished(self, success, message):
        """Handle export completion"""
        # Clean up progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Re-enable controls
        self.export_btn.setEnabled(True)
        
        # Show completion message
        if success:
            QMessageBox.information(self, "Export Complete", message)
            # self.accept()  # Close dialog on success
        else:
            QMessageBox.critical(self, "Export Failed", message)
        
        # Clean up worker
        if self.export_worker:
            self.export_worker.quit()
            self.export_worker.wait()
            self.export_worker = None
            
    def _save_current_settings(self):
        """Save current dialog settings to application settings"""
        if not self.parent_window or not hasattr(self.parent_window, 'settings_mgr'):
            return
        
        settings_mgr = self.parent_window.settings_mgr
        
        # Save current UI values to app settings
        settings_mgr.set_app_setting('export_codec', self.codec_combo.currentText())
        settings_mgr.set_app_setting('export_output_type', self.output_type_combo.currentText())
        settings_mgr.set_app_setting('export_use_input_folder', self.use_input_folder_checkbox.isChecked())
        settings_mgr.set_app_setting('export_filename_template', self.filename_template_edit.text())
        settings_mgr.set_app_setting('export_antialias', self.antialias_checkbox.isChecked())
        settings_mgr.set_app_setting('export_quantizer', self.quantizer_spin.value())
        settings_mgr.set_app_setting('export_include_original', self.include_original_checkbox.isChecked())
        settings_mgr.set_app_setting('export_multiple', self.export_multiple_checkbox.isChecked())
        settings_mgr.set_app_setting('export_folder_path', self.folder_edit.text())
        
        # Save to disk
        settings_mgr.save_app_settings()
        
        # Show confirmation
        QMessageBox.information(self, "Settings Saved", "Export settings have been saved as defaults.")
    
    def _load_saved_settings(self):
        """Load previously saved settings from application settings"""
        if not self.parent_window or not hasattr(self.parent_window, 'settings_mgr'):
            return
        
        settings_mgr = self.parent_window.settings_mgr
        
        # Load and apply saved settings
        codec = settings_mgr.get_app_setting('export_codec', 'prores')
        codec_index = self.codec_combo.findText(codec)
        if codec_index >= 0:
            self.codec_combo.setCurrentIndex(codec_index)
        
        output_type = settings_mgr.get_app_setting('export_output_type', 'Segmentation-Matte')
        output_index = self.output_type_combo.findText(output_type)
        if output_index >= 0:
            self.output_type_combo.setCurrentIndex(output_index)
        
        self.use_input_folder_checkbox.setChecked(settings_mgr.get_app_setting('export_use_input_folder', True))
        self.filename_template_edit.setText(settings_mgr.get_app_setting('export_filename_template', '{input_name}-{output_type}'))
        self.antialias_checkbox.setChecked(settings_mgr.get_app_setting('export_antialias', False))
        self.quantizer_spin.setValue(settings_mgr.get_app_setting('export_quantizer', 14))
        self.include_original_checkbox.setChecked(settings_mgr.get_app_setting('export_include_original', False))
        self.export_multiple_checkbox.setChecked(settings_mgr.get_app_setting('export_multiple', False))
        
        folder_path = settings_mgr.get_app_setting('export_folder_path', '')
        if folder_path and os.path.exists(folder_path):
            self.folder_edit.setText(folder_path)