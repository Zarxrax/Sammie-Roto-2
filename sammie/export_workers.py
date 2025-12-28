# sammie/export_workers.py
"""
Export worker threads for different export modes.
"""
import os
import av
import numpy as np
import OpenEXR
import Imath
from fractions import Fraction
from PIL import Image
from PySide6.QtCore import QThread, Signal
from sammie import sammie
from .export_formats import ExportSettings, FormatRegistry
import re


class BaseExportWorker(QThread):
    """Base worker thread for exports"""
    
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished = Signal(bool, str)
    
    def __init__(self, settings: ExportSettings, points, total_frames, parent_window=None):
        super().__init__()
        self.settings = settings
        self.points = points
        self.total_frames = total_frames
        self.should_cancel = False
        self.parent_window = parent_window
        
        # Calculate frame range
        if settings.use_inout and settings.in_point is not None and settings.out_point is not None:
            self.start_frame = max(0, settings.in_point)
            self.end_frame = min(total_frames - 1, settings.out_point)
        else:
            self.start_frame = 0
            self.end_frame = total_frames - 1
        
        self.export_frame_count = self.end_frame - self.start_frame + 1
    
    def cancel(self):
        self.should_cancel = True
    
    def _get_view_options(self, output_type: str, antialias: bool) -> dict:
        """Get view options for rendering"""
        settings_mgr = self.parent_window.settings_mgr if self.parent_window else None
        bgcolor = (0, 255, 0)
        if settings_mgr:
            bgcolor = settings_mgr.get_session_setting("bgcolor", (0, 255, 0))
        
        view_options = {'view_mode': output_type}
        
        if output_type.startswith('Segmentation-'):
            view_options['antialias'] = antialias
            if output_type == 'Segmentation-BGcolor':
                view_options['bgcolor'] = bgcolor
        elif output_type.startswith('Matting-'):
            if output_type == 'Matting-BGcolor':
                view_options['bgcolor'] = bgcolor
        elif output_type == 'ObjectRemoval':
            view_options['show_removal_mask'] = False
        
        return view_options
    
    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize name for use in filenames or layer names"""
        if not name:
            return "unnamed"
        sanitized = re.sub(r'[^\w]', '_', name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized[:20] if len(sanitized) > 20 else sanitized or "unnamed"


class VideoExportWorker(BaseExportWorker):
    """Worker for video export (single or multiple files)"""
    
    def __init__(self, settings: ExportSettings, points, total_frames, 
                 output_paths: list, object_ids: list = None, parent_window=None):
        super().__init__(settings, points, total_frames, parent_window)
        self.output_paths = output_paths  # List of output paths
        self.object_ids = object_ids or [settings.object_id]
        self.format = FormatRegistry.get_format(settings.format_id)
    
    def run(self):
        try:
            if len(self.output_paths) > 1:
                self._export_multiple()
            else:
                self._export_single(self.output_paths[0], self.object_ids[0])
                if not self.should_cancel:
                    frame_range_msg = f" (frames {self.start_frame}-{self.end_frame})" if self.settings.use_inout else ""
                    self.finished.emit(True, f"Video exported successfully{frame_range_msg} to {self.output_paths[0]}")
        except InterruptedError as e:
            # User cancelled - emit with cancel message
            self.finished.emit(False, str(e))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.finished.emit(False, f"Export failed: {e}\n\n{tb}")
    
    def _export_multiple(self):
        """Export multiple videos for different objects"""
        total_progress_steps = len(self.output_paths) * self.export_frame_count
        current_step = 0
        exported_files = []
        
        for i, (output_path, object_id) in enumerate(zip(self.output_paths, self.object_ids)):
            if self.should_cancel:
                self._cleanup_files(exported_files)
                raise InterruptedError("Export was cancelled by user")
            
            self.status_updated.emit(f"Exporting object {object_id} ({i+1}/{len(self.output_paths)})...")
            
            try:
                self._export_single(output_path, object_id, current_step, total_progress_steps)
                exported_files.append(output_path)
                current_step += self.export_frame_count
            except InterruptedError:
                # Re-raise cancel exception after cleanup
                self._cleanup_files(exported_files)
                raise
            except Exception as e:
                self._cleanup_files(exported_files)
                raise e
        
        if not self.should_cancel:
            self.status_updated.emit("All exports completed successfully")
            files_text = "\n".join([os.path.basename(f) for f in exported_files])
            frame_range_msg = f" (frames {self.start_frame}-{self.end_frame})" if self.settings.use_inout else ""
            self.finished.emit(True, f"Successfully exported {len(exported_files)} videos{frame_range_msg}:\n{files_text}")
    
    def _export_single(self, output_path: str, object_id: int, 
                      progress_offset: int = 0, total_progress_steps: int = None):
        """Export a single video file"""
        if total_progress_steps is None:
            total_progress_steps = self.export_frame_count
        
        object_id_filter = None if object_id == -1 else object_id
        has_alpha = 'Alpha' in self.settings.output_type
        
        # Create output container
        container = av.open(output_path, mode='w')
        
        # Get frame rate
        fps = sammie.VideoInfo.fps
        fps_rational = self._convert_fps_to_fraction(fps)
        
        # Create video stream
        stream = container.add_stream(self.format.get_codec_name(), rate=fps_rational)
        stream.width = sammie.VideoInfo.width
        stream.height = sammie.VideoInfo.height
        stream.pix_fmt = self.format.get_pixel_format(has_alpha)
        
        # Set codec options
        codec_options = self.format.get_codec_options(self.settings.quality)
        if has_alpha and self.format.supports_alpha:
            if self.format.format_id == 'prores':
                codec_options['profile'] = '4'
        
        for key, value in codec_options.items():
            stream.options[key] = value
        
        try:
            pts_counter = 0
            for i, frame_num in enumerate(range(self.start_frame, self.end_frame + 1)):
                if self.should_cancel:
                    container.close()
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    raise InterruptedError("Export was cancelled by user")
                
                # Get frame data
                view_options = self._get_view_options(self.settings.output_type, self.settings.antialias)
                frame_array = sammie.update_image(
                    frame_num, view_options, self.points, 
                    return_numpy=True, object_id_filter=object_id_filter
                )
                
                if frame_array is None:
                    continue
                
                # Handle alpha channel
                if has_alpha and frame_array.shape[2] != 4:
                    alpha_channel = np.full((frame_array.shape[0], frame_array.shape[1], 1), 255, dtype=np.uint8)
                    frame_array = np.concatenate([frame_array, alpha_channel], axis=2)
                elif not has_alpha and frame_array.shape[2] == 4:
                    frame_array = frame_array[:, :, :3]
                
                # Create AV frame
                av_format = 'rgba' if has_alpha else 'rgb24'
                av_frame = av.VideoFrame.from_ndarray(frame_array, format=av_format)
                av_frame.pts = pts_counter
                pts_counter += 1
                
                # Encode and write
                for packet in stream.encode(av_frame):
                    container.mux(packet)
                
                # Update progress
                current_progress_step = progress_offset + i + 1
                progress = int(current_progress_step / total_progress_steps * 100)
                self.progress_updated.emit(progress)
            
            # Flush remaining frames
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
    
    @staticmethod
    def _convert_fps_to_fraction(fps: float) -> Fraction:
        """Convert float FPS to fraction for exact representation"""
        if abs(fps - 29.97) < 0.01:
            return Fraction(30000, 1001)
        elif abs(fps - 23.976) < 0.01:
            return Fraction(24000, 1001)
        elif abs(fps - 59.94) < 0.01:
            return Fraction(60000, 1001)
        else:
            return Fraction(fps).limit_denominator()
    
    @staticmethod
    def _cleanup_files(file_paths: list):
        """Clean up partially exported files"""
        for path in file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


class SequenceExportWorker(BaseExportWorker):
    """Worker for frame sequence export (EXR, PNG)"""
    
    def __init__(self, settings: ExportSettings, points, total_frames, 
                 base_filename: str, parent_window=None):
        super().__init__(settings, points, total_frames, parent_window)
        self.base_filename = base_filename
        self.format = FormatRegistry.get_format(settings.format_id)
    
    def run(self):
        try:
            if self.format.format_id == 'exr':
                self._export_exr_sequence()
            elif self.format.format_id == 'png':
                self._export_png_sequence()
            else:
                raise ValueError(f"Unsupported sequence format: {self.format.format_id}")
        except InterruptedError as e:
            # User cancelled - emit with cancel message
            self.finished.emit(False, str(e))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.finished.emit(False, f"Export failed: {e}\n\n{tb}")
    
    def _export_exr_sequence(self):
        """Export EXR frame sequence with multiple object layers"""
        output_dir = self.settings.output_dir
        
        # Get all unique object IDs
        all_object_ids = sorted(set(point['object_id'] for point in self.points))
        if not all_object_ids:
            self.finished.emit(False, "No objects found to export")
            return
        
        # Get object names
        object_names = {}
        if self.parent_window and hasattr(self.parent_window, 'settings_mgr'):
            object_names = self.parent_window.settings_mgr.get_session_setting("object_names", {})
        
        exported_files = []
        
        for i, frame_num in enumerate(range(self.start_frame, self.end_frame + 1)):
            if self.should_cancel:
                self._cleanup_files(exported_files)
                raise InterruptedError("Export was cancelled by user")
            
            self.status_updated.emit(f"Exporting frame {frame_num + 1}/{self.end_frame + 1}...")
            
            frame_filename = f"{self.base_filename}.{frame_num:04d}.exr"
            frame_path = os.path.join(output_dir, frame_filename)
            
            try:
                exr_data = {}
                view_options = self._get_view_options(self.settings.output_type, self.settings.antialias)
                
                # Export each object as a layer
                for obj_id in all_object_ids:
                    mask_array = sammie.update_image(
                        frame_num, view_options, self.points,
                        return_numpy=True, object_id_filter=obj_id
                    )
                    
                    if mask_array is not None:
                        # Convert to grayscale if needed
                        if len(mask_array.shape) == 3 and mask_array.shape[2] > 1:
                            mask_array = mask_array[:, :, 0]
                        
                        # Normalize to 0-1 range
                        if mask_array.dtype == np.uint8:
                            mask_array = mask_array.astype(np.float32) / 255.0
                        elif mask_array.dtype != np.float32:
                            mask_array = mask_array.astype(np.float32)
                        
                        # Generate layer name
                        object_name = object_names.get(str(obj_id), "")
                        if object_name:
                            sanitized_name = self._sanitize_name(object_name)
                            layer_name = f'{obj_id}_{sanitized_name}'
                        else:
                            layer_name = f'{obj_id}'
                        
                        exr_data[layer_name] = mask_array
                
                # Include original frame if requested
                if self.settings.include_original:
                    original_view_options = {'view_mode': 'None'}
                    original_array = sammie.update_image(
                        frame_num, original_view_options, self.points,
                        return_numpy=True, object_id_filter=None
                    )
                    
                    if original_array is not None:
                        if original_array.dtype == np.uint8:
                            original_array = original_array.astype(np.float32) / 255.0
                        elif original_array.dtype != np.float32:
                            original_array = original_array.astype(np.float32)
                        
                        # Split RGB channels
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
                
                progress = int((i + 1) / self.export_frame_count * 100)
                self.progress_updated.emit(progress)
                
            except InterruptedError:
                # Re-raise cancel exception
                raise
            except Exception as e:
                print(f"Error exporting frame {frame_num + 1}: {e}")
                self._cleanup_files(exported_files)
                raise e
        
        if not self.should_cancel:
            self.status_updated.emit("EXR sequence export completed")
            frame_range_msg = f" (frames {self.start_frame}-{self.end_frame})" if self.settings.use_inout else ""
            self.finished.emit(True, f"Successfully exported {len(exported_files)} EXR frames{frame_range_msg} to {output_dir}")
    
    def _export_png_sequence(self):
        """Export PNG frame sequence"""
        output_dir = self.settings.output_dir
        object_id_filter = None if self.settings.object_id == -1 else self.settings.object_id
        has_alpha = 'Alpha' in self.settings.output_type
        
        exported_files = []
        
        for i, frame_num in enumerate(range(self.start_frame, self.end_frame + 1)):
            if self.should_cancel:
                self._cleanup_files(exported_files)
                raise InterruptedError("Export was cancelled by user")
            
            self.status_updated.emit(f"Exporting frame {frame_num + 1}/{self.end_frame + 1}...")
            
            frame_filename = f"{self.base_filename}.{frame_num:04d}.png"
            frame_path = os.path.join(output_dir, frame_filename)
            
            try:
                # Get frame data
                view_options = self._get_view_options(self.settings.output_type, self.settings.antialias)
                frame_array = sammie.update_image(
                    frame_num, view_options, self.points,
                    return_numpy=True, object_id_filter=object_id_filter
                )
                
                if frame_array is not None:
                    # Handle alpha channel
                    if has_alpha and frame_array.shape[2] != 4:
                        alpha_channel = np.full((frame_array.shape[0], frame_array.shape[1], 1), 255, dtype=np.uint8)
                        frame_array = np.concatenate([frame_array, alpha_channel], axis=2)
                    elif not has_alpha and frame_array.shape[2] == 4:
                        frame_array = frame_array[:, :, :3]
                    
                    # Save as PNG using PIL
                    mode = 'RGBA' if has_alpha else 'RGB'
                    img = Image.fromarray(frame_array, mode=mode)
                    img.save(frame_path, 'PNG')
                    exported_files.append(frame_path)
                
                progress = int((i + 1) / self.export_frame_count * 100)
                self.progress_updated.emit(progress)
                
            except InterruptedError:
                # Re-raise cancel exception
                raise
            except Exception as e:
                print(f"Error exporting frame {frame_num + 1}: {e}")
                self._cleanup_files(exported_files)
                raise e
        
        if not self.should_cancel:
            self.status_updated.emit("PNG sequence export completed")
            frame_range_msg = f" (frames {self.start_frame}-{self.end_frame})" if self.settings.use_inout else ""
            self.finished.emit(True, f"Successfully exported {len(exported_files)} PNG frames{frame_range_msg} to {output_dir}")
    
    @staticmethod
    def _write_exr_file(filepath: str, data_dict: dict):
        """Write EXR file with multiple layers"""
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
    
    @staticmethod
    def _cleanup_files(file_paths: list):
        """Clean up partially exported files"""
        for path in file_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
