import cv2
import os
import numpy as np
import torch
import gc
from tqdm import tqdm
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt
from sammie import core
from sammie.settings_manager import get_settings_manager
from sammie.model_downloader import ensure_models

from matting.base import MattingManager
from matting.matanyone.downloads import DOWNLOADS

class MatAnyManager(MattingManager):
    """Matting manager that uses the MatAnyone / MatAnyone2 model."""

    def __init__(self, model_id):
        super().__init__()
        self.BACKEND = model_id

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        """Load the MatAnyone model and return processor"""
        from matting.matanyone.vendor.inference.inference_core import InferenceCore
        from matting.matanyone.vendor.utils.get_default_model import get_matanyone_model

        device = self._prepare_device(load_to_cpu)
        settings_mgr = get_settings_manager()
        matting_model = self.BACKEND
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        combined = settings_mgr.get_session_setting("matany_combined", False)

        if matting_model == "MatAnyone2":
            model_spec = DOWNLOADS["matanyone2"]
            checkpoint = str(model_spec.final_path)
            if not ensure_models(model_spec, parent=parent_window):
                return False  # user cancelled or download failed
        else:
            model_spec = DOWNLOADS["matanyone"]
            checkpoint = str(model_spec.final_path)
            if not ensure_models(model_spec, parent=parent_window):
                return False  # user cancelled or download failed

        matanyone = get_matanyone_model(checkpoint, device=device)
        print(f"Loaded {matting_model} model to {device} with max size {max_size} and combined={combined}")

        # Initialize inference processor
        self.processor = InferenceCore(matanyone, cfg=matanyone.cfg, device=device)
        return self.processor

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        """
        Run matting on all frames, using multiple keyframes for each object.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        if self.processor is None:
            print("Matting model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        device = core.DeviceManager.get_device()
        frame_count = core.VideoInfo.total_frames

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get unique object IDs from points list
        object_ids = sorted(list(set(point['object_id'] for point in points_list if 'object_id' in point)))
        if not object_ids:
            print("No objects found for matting")
            return 0

        # Find all keyframes for each object (within processing range)
        object_keyframes = {}
        for object_id in object_ids:
            keyframes = sorted(list(set(
                point['frame'] for point in points_list
                if point.get('object_id') == object_id and start_frame <= point['frame'] <= end_frame
            )))
            if keyframes:
                object_keyframes[object_id] = keyframes
            else:
                print(f"No frames found for object {object_id} in range {start_frame}-{end_frame}")

        if not object_keyframes:
            print("No valid keyframes found for any objects in the specified range")
            return 0

        # When combined mode is requested, run a single pass using the union of all
        # object masks loaded in memory — no files are written to disk.
        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            earliest_keyframe = min(kf[0] for kf in object_keyframes.values())
            object_ids = [0]
            object_keyframes = {0: [earliest_keyframe]}
        else:
            combine_ids = None

        # Calculate total operations for progress tracking
        total_operations = 0
        for object_id, keyframes in object_keyframes.items():
            first_keyframe = keyframes[0]

            # Operations before first keyframe (backward propagation to start_frame)
            total_operations += first_keyframe - start_frame

            # Operations between keyframes and after last keyframe to end_frame
            for i in range(len(keyframes)):
                if i == len(keyframes) - 1:
                    total_operations += end_frame - keyframes[i] + 1
                else:
                    total_operations += keyframes[i + 1] - keyframes[i]

        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations)

        # Create matting directory if it doesn't exist
        os.makedirs(core.matting_dir, exist_ok=True)

        # If combined mode is selected, delete any existing matting files except object 0.
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        images = self._collect_image_paths(start_frame, end_frame)

        operations_completed = 0

        # Process each object with its keyframes
        for object_id, keyframes in object_keyframes.items():
            if progress_dialog.wasCanceled():
                break

            pbar.set_description(f"Object {object_id}")

            # Process segments for this object
            success = self._process_object_with_keyframes(
                images, object_id, keyframes, end_frame + 1, device,
                progress_dialog, operations_completed, total_operations, pbar, parent_window,
                start_frame=start_frame, combine_ids=combine_ids
            )

            if not success:
                break

            # Update operations completed for this object
            first_keyframe = keyframes[0]
            operations_completed += first_keyframe - start_frame  # backward from first keyframe

            for i in range(len(keyframes)):
                if i == len(keyframes) - 1:
                    operations_completed += end_frame - keyframes[i] + 1  # last keyframe to end
                else:
                    operations_completed += keyframes[i + 1] - keyframes[i]  # between keyframes

        # Close tqdm progress bar
        pbar.close()

        # Final cleanup
        core.DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(100)
            if frame_count == frames_to_process:
                self.propagated = True  # only set propagated to True if the entire video was processed
            else:
                self.propagated = False
            print("Matting completed")
            self._notify('matting_complete')
            return 1

    def _process_object_with_keyframes(self, images, object_id, keyframes, frame_count, device, progress_dialog,
                                       operations_completed, total_operations, pbar, parent_window, start_frame=0,
                                       combine_ids=None):
        """
        Process a single object using multiple keyframes.

        Args:
            images: List of image paths
            object_id: ID of the object to process (also the output file label)
            keyframes: Sorted list of keyframe indices for this object
            frame_count: Total number of frames
            device: Processing device
            progress_dialog: Progress dialog for user feedback
            operations_completed: Number of operations completed so far
            total_operations: Total operations for all objects
            pbar: tqdm progress bar
            parent_window: Parent window
            start_frame: Starting frame for the processing range
            combine_ids: If set, union masks for these IDs in memory rather than
                         loading a single object mask from disk

        Returns:
            bool: True if successful, False if cancelled or failed
        """
        first_keyframe = keyframes[0]

        # Load and validate the first keyframe mask
        mask, original_size = self._load_mask_for_matting(object_id, first_keyframe, device,
                                                          combine_ids=combine_ids)
        if mask is None:
            return False

        # Special case for single frame
        if len(images) == 1:
            return self._process_single_frame(images[0], mask, object_id, original_size, device, frame_number=start_frame)

        current_operations = operations_completed

        # 1. Process backward from first keyframe to start_frame
        if first_keyframe > start_frame:
            success = self._process_backward(images, mask, object_id, first_keyframe,
                                             original_size, device, progress_dialog, current_operations,
                                             total_operations, parent_window, pbar, start_frame_offset=start_frame)
            if not success:
                return False
            current_operations += first_keyframe - start_frame

        # 2. Process forward segments between keyframes
        for i in range(len(keyframes)):
            if progress_dialog.wasCanceled():
                return False

            current_keyframe = keyframes[i]

            # Load mask for current keyframe (refresh for each segment)
            mask, original_size = self._load_mask_for_matting(object_id, current_keyframe, device,
                                                              combine_ids=combine_ids)
            if mask is None:
                print(f"Failed to load mask for object {object_id} at keyframe {current_keyframe}")
                return False

            # Determine end frame for this segment
            if i == len(keyframes) - 1:
                end_frame = frame_count  # Last keyframe - process to end of range
            else:
                end_frame = keyframes[i + 1]  # Process to next keyframe (exclusive)

            # Process this forward segment
            if end_frame > current_keyframe:
                success = self._process_forward(images, mask, object_id, current_keyframe, original_size,
                                                device, progress_dialog, current_operations, total_operations,
                                                parent_window, end_frame, pbar, start_frame_offset=start_frame)
                if not success:
                    return False
                current_operations += end_frame - current_keyframe

        return True

    def _process_single_frame(self, frame_path, mask, object_id, original_size, device, frame_number=0):
        """Process a single frame for matting"""
        try:
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._resize_image(img)
            img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

            output_prob = self.processor.step(img, mask, objects=[1])
            for i in range(10):  # Warmup iterations
                output_prob = self.processor.step(img, first_frame_pred=True)
                core.DeviceManager.clear_cache()

            mat = self.processor.output_prob_to_mask(output_prob)
            mat = mat.detach().cpu().numpy()
            mat = (mat * 255).astype(np.uint8)
            mat = self._restore_image_size(mat, original_size)

            mat_filename = os.path.join(core.matting_dir, f"{frame_number:05d}", f"{object_id}.png")
            os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
            cv2.imwrite(mat_filename, mat)
            return True

        except Exception as e:
            print(f"Error processing single frame: {e}")
            return False

    def _process_forward(self, images, mask, object_id, start_frame, original_size, device, progress_dialog,
                         operations_completed, total_operations, parent_window, end_frame=None, pbar=None,
                         start_frame_offset=0):
        """
        Process frames forward from start_frame.

        Args:
            images: List of image paths
            mask: Initial mask tensor
            object_id: Object ID
            start_frame: Starting frame (inclusive)
            original_size: Original image size
            device: Processing device
            progress_dialog: Progress dialog
            operations_completed: Operations completed before this segment
            total_operations: Total operations
            parent_window: Parent window
            end_frame: Ending frame (exclusive). If None, process to end of images.
            pbar: tqdm progress bar
            start_frame_offset: Offset for mapping array indices to absolute frame numbers

        Returns:
            bool: True if successful, False if cancelled or failed
        """
        if end_frame is None:
            end_frame = start_frame + len(images)

        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        try:
            for frame_number in range(start_frame, end_frame):
                if progress_dialog.wasCanceled():
                    return False

                # Map absolute frame number to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    print(f"Warning: Frame {frame_number} out of range for images array")
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

                if frame_number == start_frame:
                    # First frame - initialize with mask
                    output_prob = self.processor.step(img, mask, objects=[1])
                    for i in range(10):  # Warmup iterations
                        output_prob = self.processor.step(img, first_frame_pred=True)
                        core.DeviceManager.clear_cache()
                else:
                    # Subsequent frames - propagate
                    output_prob = self.processor.step(img)

                # Convert to matte
                mat = self.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(core.matting_dir, f"{frame_number:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                core.DeviceManager.clear_cache()

                # Update display at the specified frequency
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                # Update progress
                if pbar is not None:
                    pbar.update(1)
                current_progress = int(((operations_completed + (frame_number - start_frame) + 1) * 100) / total_operations)
                progress_dialog.setValue(current_progress)
                QApplication.processEvents()

            return True

        except Exception as e:
            print(f"Error in forward processing: {e}")
            return False

    def _process_backward(self, images, mask, object_id, start_frame, original_size, device, progress_dialog,
                          operations_completed, total_operations, parent_window, pbar=None, start_frame_offset=0):
        """Process frames backward from start_frame"""

        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        try:
            for frame_number in range(start_frame, start_frame_offset - 1, -1):
                if progress_dialog.wasCanceled():
                    return False

                # Map absolute frame number to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    print(f"Warning: Frame {frame_number} out of range for images array")
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

                if frame_number == start_frame:
                    # First frame - initialize with mask
                    output_prob = self.processor.step(img, mask, objects=[1])
                    for i in range(10):  # Warmup iterations
                        output_prob = self.processor.step(img, first_frame_pred=True)
                        core.DeviceManager.clear_cache()
                else:
                    # Subsequent frames - propagate
                    output_prob = self.processor.step(img)

                # Convert to matte
                mat = self.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(core.matting_dir, f"{frame_number:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                core.DeviceManager.clear_cache()

                # Update display at the specified frequency
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                # Update progress
                if pbar is not None:
                    pbar.update(1)
                operations_completed += 1
                progress_dialog.setValue(operations_completed * 100 // total_operations)
                QApplication.processEvents()

            return True

        except Exception as e:
            print(f"Error in backward processing: {e}")
            return False
