# sammie/matting.py
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


class MattingManager:
    """
    Shared base class for matting managers.
    Provides common infrastructure: callbacks, image resize/restore, mask loading,
    progress dialog helpers, and matting directory management.
    Subclasses must implement load_matting_model() and run_matting().
    """

    def __init__(self):
        self.processor = None
        self.propagated = False  # whether we have propagated the mattes
        self.callbacks = []

    def add_callback(self, callback):
        """Add callback for matting events"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except RuntimeError as e:
                # Allow cancellation to propagate
                if str(e) == "USER_CANCELLED":
                    raise
                print(f"Callback error: {e}")
            except Exception as e:
                print(f"Callback error: {e}")

    def _prepare_device(self, load_to_cpu=False):
        """Return the appropriate torch device and clear cache"""
        core.DeviceManager.clear_cache()
        if load_to_cpu:
            return torch.device('cpu')
        return core.DeviceManager.get_device()

    def unload_matting_model(self):
        """Unload the matting model and clear cache"""
        self.processor = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded Matting model")

    def _resize_image(self, image):
            """Resize image and ensure dimensions are multiples of 8 for the model."""
            settings_mgr = get_settings_manager()
            max_size = settings_mgr.get_session_setting("matany_res", 0)
            h, w = image.shape[:2]
            
            # 1. Determine scaling factor
            scale = 1.0
            if max_size > 0:
                min_side = min(h, w)
                if min_side > max_size:
                    scale = max_size / min_side

            # 2. ALWAYS round to a multiple of 8
            # This ensures [3, 1384, 600] instead of [3, 1390, 602]
            new_h = (int(h * scale) // 8) * 8
            new_w = (int(w * scale) // 8) * 8
            
            # 3. Always resize, even if scale is 1.0, to catch those extra pixels
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def _restore_image_size(self, image, original_size):
        """Restore image to original size. original_size must be (w, h) as expected by cv2."""
        original_w, original_h = original_size
        restored_image = cv2.resize(image, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        return restored_image

    def _load_mask_for_matting(self, object_id, frame_number, device, combine_ids=None):
        """
        Load and validate a mask for matting processing.

        If combine_ids is provided, masks for all IDs in that list are unioned
        in memory and returned as a single mask, without touching any files on disk.
        object_id is used only as the label for error messages in that case.

        Args:
            object_id: ID of the object (or output label when combining)
            frame_number: Frame number
            device: Processing device
            combine_ids: Optional list of object IDs to union into a single mask

        Returns:
            tuple: (mask_tensor, original_size) or (None, None) if failed
        """
        if combine_ids:
            union_mask = None
            original_size = None
            for oid in combine_ids:
                mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{oid}.png")
                if not os.path.exists(mask_filename):
                    continue
                m = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if original_size is None:
                    original_size = m.shape[1::-1]
                union_mask = m if union_mask is None else np.maximum(union_mask, m)
            if union_mask is None or not np.any(union_mask):
                print(f"Combined mask is blank or missing for frame {frame_number}")
                return None, None
            mask = self._resize_image(union_mask)
            mask = torch.tensor(mask, dtype=torch.float32, device=device)
            return mask, original_size

        mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{object_id}.png")
        if not os.path.exists(mask_filename):
            print(f"Mask not found for object {object_id} at frame {frame_number}: {mask_filename}")
            return None, None

        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            print(f"Mask is blank or invalid for object {object_id} at frame {frame_number}")
            return None, None

        original_size = mask.shape[1::-1]
        mask = core.apply_mask_postprocessing(mask)
        mask = self._resize_image(mask)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)

        return mask, original_size

    def _make_progress_dialog(self, parent_window, total_operations, unit="frame"):
        """Create and show the Qt progress dialog and tqdm bar"""
        progress_dialog = QProgressDialog("Running matting...", "Cancel", 0, 100, parent_window)
        progress_dialog.setWindowTitle("Matting Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()
        pbar = tqdm(total=total_operations, desc="Matting Progress", unit=unit)
        return progress_dialog, pbar

    def _get_frame_range(self):
        """
        Read in/out points from settings and return (start_frame, end_frame, frames_to_process).
        """
        settings_mgr = get_settings_manager()
        frame_count = core.VideoInfo.total_frames
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1
        return start_frame, end_frame, frames_to_process

    def _collect_image_paths(self, start_frame, end_frame):
        """Return a list of existing frame image paths in [start_frame, end_frame]."""
        extension = core.get_frame_extension()
        images = []
        for frame_number in range(start_frame, end_frame + 1):
            image_filename = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")
            if os.path.exists(image_filename):
                images.append(image_filename)
        return images

    def clear_matting(self):
        """Clear matting data"""
        import shutil
        if os.path.exists(core.matting_dir):
            shutil.rmtree(core.matting_dir)
        os.makedirs(core.matting_dir)
        self.propagated = False
        print("Matting data cleared")

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        raise NotImplementedError("Subclasses must implement load_matting_model()")

    def run_matting(self, points_list, parent_window, combined=False):
        raise NotImplementedError("Subclasses must implement run_matting()")


# ---------------------------------------------------------------------------
# MatAnyone backend
# ---------------------------------------------------------------------------

class MatAnyManager(MattingManager):
    """Matting manager that uses the MatAnyone / MatAnyone2 model."""

    BACKEND = "matanyone"

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        """Load the MatAnyone model and return processor"""
        from matanyone.inference.inference_core import InferenceCore
        from matanyone.utils.get_default_model import get_matanyone_model

        device = self._prepare_device(load_to_cpu)
        settings_mgr = get_settings_manager()
        matting_model = settings_mgr.get_session_setting("matany_model", "MatAnyone2")
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        combined = settings_mgr.get_session_setting("matany_combined", False)

        if matting_model == "MatAnyone2":
            checkpoint = "./checkpoints/matanyone2.pth"
            if not ensure_models("matanyone2", parent=parent_window):
                return False  # user cancelled or download failed
        else:
            checkpoint = "./checkpoints/matanyone.pth"
            if not ensure_models("matanyone", parent=parent_window):
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
            return self._process_single_frame(images[0], mask, object_id, original_size, device)

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

    def _process_single_frame(self, frame_path, mask, object_id, original_size, device):
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

            mat_filename = os.path.join(core.matting_dir, f"00000", f"{object_id}.png")
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


# ---------------------------------------------------------------------------
# VideoMaMa backend
# ---------------------------------------------------------------------------

class VideoMaMaManager(MattingManager):
    """
    Matting manager that uses the VideoMaMa model.

    """

    BACKEND = "videomama"

    def unload_matting_model(self):
        """Unload the VideoMaMa pipeline and free VRAM"""
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, 'unet'):
                    self.pipeline.unet = None
                if hasattr(self.pipeline, 'vae'):
                    self.pipeline.vae = None
            except Exception as e:
                print(f"Warning during pipeline teardown: {e}")
            self.pipeline = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded VideoMaMa model")
        
    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        """
        Load the VideoMaMa pipeline.
        """
        from videomama.pipeline_svd_mask_numpy import VideoInferencePipeline

        device = self._prepare_device(load_to_cpu)
        settings_mgr = get_settings_manager()
        matting_model = settings_mgr.get_session_setting("matany_model", "VideoMaMa")
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        overlap = settings_mgr.get_session_setting("matany_overlap", 2)
        batch_size = settings_mgr.get_session_setting("matany_chunk", 16)
        combined = settings_mgr.get_session_setting("matany_combined", False)

        if not ensure_models(["videomama", "svd_vae"], parent=parent_window):
            return False  # user cancelled or download failed

        try:
            self.pipeline = VideoInferencePipeline(
                base_model_path=os.path.join("checkpoints", "videomama"),
                unet_checkpoint_path=os.path.join("checkpoints", "videomama"),
                weight_dtype=torch.float16,
                device=str(device),
                enable_model_cpu_offload=False,    # Not much benefit here, since the vae is a small model
                vae_encode_chunk_size=1,          # Process VAE in small chunks, increasing doesnt help anything
                attention_mode="auto",            # Use xformers if available, else SDPA
                enable_vae_tiling=False,        # Tiling VAE is not worth it
                enable_vae_slicing=True,          # Process VAE one image at a time
            )
            print(f"Loaded {matting_model} model to {device} with max size {max_size}, overlap={overlap}, batch={batch_size}, and combined={combined}")
            return True

        except Exception as e:
            print(f"Error loading VideoMaMa pipeline: {e}")
            self.pipeline = None
            return False

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        """
        Run matting on all frames, using tracked segmentation data.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog
            combined (bool): If True, union all object masks into a single combined mask
                and run matting in a single pass instead of once per object.

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """

        if self.pipeline is None:
            print("VideoMaMa model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        frame_count = core.VideoInfo.total_frames
        extension = core.get_frame_extension()
        settings_mgr = get_settings_manager()
        
        # --- VideoMaMa batch settings ---
        batch_size = settings_mgr.get_session_setting("matany_chunk", 16)   # frames per chunk sent to the model
        overlap = settings_mgr.get_session_setting("matany_overlap", 2)     # frames re-processed at each boundary for continuity
        if overlap == 0: 
            enable_boundary_blend = False
        else: 
            enable_boundary_blend = True # blend overlap frames linearly at chunk boundaries

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get unique object IDs from points list
        object_ids = sorted(list(set(point['object_id'] for point in points_list if 'object_id' in point)))
        if not object_ids:
            print("No objects found for matting")
            return 0

        # When combined mode is requested, run a single pass using the union of all
        # object masks loaded in memory — no files are written to disk.
        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            object_ids = [0]
        else:
            combine_ids = None

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

        # Calculate total operations for progress tracking
        total_operations = len(self._generate_windows(frames_to_process, batch_size, overlap)) * len(object_ids)

        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations, unit="batch")
        operations_completed = 0

        try:
            # Process each object separately
            for object_id in object_ids:
                if progress_dialog.wasCanceled():
                    break

                pbar.set_description(f"Object {object_id}")
                print(f"Processing object {object_id}...")

                batches_completed = self._process_object(
                    object_id, start_frame, end_frame, overlap, batch_size, extension,
                    progress_dialog, pbar, operations_completed,
                    total_operations, parent_window, combine_ids=combine_ids,
                    enable_boundary_blend=enable_boundary_blend
                )

                if batches_completed is None:  # cancelled or failed
                    break

                operations_completed += batches_completed

        except Exception as e:
            pbar.close()
            progress_dialog.close()
            raise

        # Final cleanup
        pbar.close()
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

    def _load_batch(self, abs_start, abs_end, object_id, extension,
                    combine_ids=None, crop_rect=None):
        """
        Load a single batch window of frames and masks from disk, resized using
        the same proportional downscaling as MatAnyone (matany_res setting).
 
        If combine_ids is provided, masks for all IDs in that list are unioned
        in memory per frame rather than loading a single object mask from disk.
 
        If crop_rect is provided, both frames and masks are cropped to that region
        before being passed through _resize_image, so the model only processes the
        relevant portion of the frame.

        Args:
            abs_start:   Absolute start frame (inclusive)
            abs_end:     Absolute end frame (exclusive)
            object_id:   Object ID for mask lookup (ignored when combine_ids is set)
            extension:   Frame file extension
            combine_ids: Optional list of object IDs to union into a single mask
            crop_rect:   Optional (x1, y1, x2, y2) from core.compute_mask_bounding_box.
                         When set, frames and masks are cropped before resizing.
 
        Returns:
            tuple: (cond_frames, mask_frames, valid)
                - cond_frames: list of np.ndarray RGB uint8 (H, W, 3)
                - mask_frames: list of np.ndarray grayscale uint8 (H, W), binarized
                - valid: False if any source frame is missing
        """
        cond_frames = []
        mask_frames = []
 
        for frame_num in range(abs_start, abs_end):
            frame_path = os.path.join(core.frames_dir, f"{frame_num:05d}.{extension}")
            if not os.path.exists(frame_path):
                print(f"Warning: Frame not found: {frame_path}")
                return [], [], False
 
            # Load and convert frame to RGB, then proportionally downscale via matany_res
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop_rect is not None:
                frame = core.apply_crop(frame, crop_rect)
            frame = self._resize_image(frame)
            resized_h, resized_w = frame.shape[:2]
 
            # Load mask — union across all combine_ids in memory, or single object from disk
            if combine_ids:
                union_mask = None
                for oid in combine_ids:
                    mask_path = os.path.join(core.mask_dir, f"{frame_num:05d}", f"{oid}.png")
                    if not os.path.exists(mask_path):
                        continue
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    union_mask = m if union_mask is None else np.maximum(union_mask, m)
                if union_mask is not None:
                    union_mask = core.apply_mask_postprocessing(union_mask)
                    if crop_rect is not None:
                        union_mask = core.apply_crop(union_mask, crop_rect)
                    mask = cv2.resize(union_mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
            else:
                mask_path = os.path.join(core.mask_dir, f"{frame_num:05d}", f"{object_id}.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = core.apply_mask_postprocessing(mask)
                    if crop_rect is not None:
                        mask = core.apply_crop(mask, crop_rect)
                    mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
 
            # Binarize mask
            mask = (mask > 127).astype(np.uint8) * 255
 
            cond_frames.append(frame)
            mask_frames.append(mask)
 
        return cond_frames, mask_frames, True

    def _generate_windows(self, total_frames, batch_size, overlap):
        """
        Generate sliding window (start, end) pairs covering total_frames.
        end is exclusive. Overlap must be less than batch_size.

        Args:
            total_frames: Total frames to cover
            batch_size: Frames per batch 
            overlap: Overlap frames between consecutive batches

        Returns:
            list of (start, end) tuples (relative indices, end exclusive)
        """
        step = batch_size - overlap
        if step <= 0:
            print(f"Warning: overlap ({overlap}) >= batch_size ({batch_size}), clamping to batch_size - 1")
            overlap = batch_size - 1
            step = 1

        if total_frames <= batch_size:
            return [(0, total_frames)]

        windows = []
        pos = 0
        while pos < total_frames:
            end = min(pos + batch_size, total_frames)
            windows.append((pos, end))
            if end >= total_frames:
                break
            pos += step

        return windows


    def _process_object(self, object_id, start_frame, end_frame, overlap, batch_size, extension,
                        progress_dialog, pbar, operations_completed, total_operations, parent_window,
                        combine_ids=None, enable_boundary_blend=True):
        """
        Process a single object across all frames using windowed batching.
 
        Args:
            object_id: Object ID to process (also the output file label)
            start_frame: First frame (inclusive)
            end_frame: Last frame (inclusive)
            overlap: Overlap frames between batches
            batch_size: Number of frames per chunk sent to the model
            extension: Frame file extension
            progress_dialog: Qt progress dialog
            pbar: tqdm progress bar
            operations_completed: Batches completed before this object (for progress math)
            total_operations: Total batch count across all objects
            parent_window: Parent window for display updates
            combine_ids: If set, union masks for these IDs in memory rather than
                         loading a single object mask from disk
            enable_boundary_blend: Linearly blend overlap frames at chunk boundaries
 
        Returns:
            int: Number of batches completed, or None if cancelled/failed
        """
        frames_to_process = end_frame - start_frame + 1
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
 
        # Original frame dimensions for restoring output
        first_frame_path = os.path.join(core.frames_dir, f"{start_frame:05d}.{extension}")
        first_frame_img = cv2.imread(first_frame_path)
        if first_frame_img is not None:
            original_h, original_w = first_frame_img.shape[:2]
        else:
            original_w = core.VideoInfo.width
            original_h = core.VideoInfo.height

        # Compute a single crop rect covering all mask extents across the full frame range.
        # This is done once before any batch processing so every batch uses an identical
        # spatial region, keeping output consistent across chunk boundaries.
        object_ids_for_bbox = combine_ids if combine_ids is not None else [object_id]
        progress_dialog.setLabelText(f"Object {object_id}: Finding region of interest...")
        QApplication.processEvents()

        crop_rect = core.compute_mask_bounding_box(
            frame_range=range(start_frame, end_frame + 1),
            object_ids=object_ids_for_bbox,
            combine_ids=combine_ids,
        )
        if crop_rect is not None:
            cx1, cy1, cx2, cy2 = crop_rect
            print(f"Object {object_id}: crop rect ({cx1}, {cy1}) -> ({cx2}, {cy2})  "
                  f"[{cx2 - cx1 + 1}x{cy2 - cy1 + 1} of {original_w}x{original_h}]")
        else:
            print(f"Object {object_id}: no mask content found, processing full frame")
 
        windows = self._generate_windows(frames_to_process, batch_size, overlap)
        batches_completed = 0

        # Stores soft alpha mattes (model working resolution, grayscale uint8) for the
        # last `overlap` frames of each batch.  Fed back as mask_frames for the overlap
        # frames of the next batch, giving the model continuity across chunk boundaries.
        # Kept soft (not binarized) so the model sees graduated edge information.
        previous_overlap_masks = None

        # Stores the last `overlap` committed alpha mattes at original resolution,
        # used for linear blending across the boundary after each non-first batch.
        prev_boundary_alphas = None
 
        for batch_idx, (window_start, window_end) in enumerate(windows):
            if progress_dialog.wasCanceled():
                return None
 
            abs_start = start_frame + window_start
            abs_end = start_frame + window_end  # exclusive
 
            # For non-first batches, the leading overlap frames are warm-up context only —
            # their output is discarded in favour of the already-committed result from the
            # previous batch. Only the non-overlap tail is saved.
            is_first_batch = (batch_idx == 0)
            output_start_offset = 0 if is_first_batch else overlap
 
            cond_frames, mask_frames, valid = self._load_batch(
                abs_start, abs_end, object_id, extension,
                combine_ids=combine_ids, crop_rect=crop_rect
            )
 
            if not valid:
                print(f"Warning: Skipping batch {batch_idx} for object {object_id} "
                    f"(frames {abs_start}-{abs_end - 1}) — missing data")
                pbar.update(1)
                batches_completed += 1
                # Reset both carry-overs so stale data is never injected after a gap
                previous_overlap_masks = None
                prev_boundary_alphas = None
                continue
 
            # --- FEEDBACK LOOP INJECTION ---
            # Replace the SAM2 masks for the overlap frames with the soft alpha mattes
            # predicted by the previous batch.  The model sees its own prior output as
            # guidance, encouraging consistent alpha values across the chunk boundary.
            if not is_first_batch and previous_overlap_masks is not None:
                for i in range(min(overlap, len(mask_frames), len(previous_overlap_masks))):
                    mask_frames[i] = previous_overlap_masks[i]
            # -------------------------------

            def _on_pipeline_progress(step, total, desc):
                progress_dialog.setLabelText(f"Batch {batch_idx + 1}/{len(windows)} — {desc}")
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise RuntimeError("USER_CANCELLED")
 
            try:
                with torch.amp.autocast('cuda', enabled=False):
                    output_frames = self.pipeline.run(
                        cond_frames=cond_frames,
                        mask_frames=mask_frames,
                        seed=42,
                        progress_callback=_on_pipeline_progress,
                    )
            except RuntimeError as e:
                if str(e) == "USER_CANCELLED":
                    return None  # signals cancel upstream
                raise
            except Exception as e:
                print(f"Error in VideoMaMa inference for batch {batch_idx}: {e}")
                raise
 
            # --- CAPTURE SOFT MASKS FOR NEXT BATCH ---
            # Store the last `overlap` frames of the current output at model resolution
            # (before _restore_image_size) as soft grayscale — NOT binarized, so the
            # model receives graduated edge values rather than a hard binary boundary.
            previous_overlap_masks = []
            for frame_out in output_frames[-overlap:]:
                alpha_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2GRAY)
                previous_overlap_masks.append(alpha_out)
            # -----------------------------------------
 
            # Discard leading overlap output on non-first batches
            committed_output = output_frames[output_start_offset:]

            # --- LINEAR BLEND AT CHUNK BOUNDARIES ---
            # The overlap frames were re-processed by the new batch, giving us a second
            # prediction for those already-committed frames (output_frames[0:overlap]).
            # We linearly blend the original committed alpha (prev_boundary_alphas) with
            # the new prediction across the overlap window and re-write those frames.
            if enable_boundary_blend and not is_first_batch and prev_boundary_alphas is not None:
                for i in range(min(overlap, len(prev_boundary_alphas), len(output_frames))):
                    new_alpha = cv2.cvtColor(output_frames[i], cv2.COLOR_RGB2GRAY)
                    new_alpha = self._restore_image_size(new_alpha, (cx2 - cx1 + 1, cy2 - cy1 + 1) if crop_rect else (original_w, original_h))
                    if crop_rect is not None:
                        new_alpha = core.expand_to_full(new_alpha, crop_rect, original_w, original_h)
                    new_weight = (i + 1) / (overlap + 1)
                    blended = (
                        (1.0 - new_weight) * prev_boundary_alphas[i].astype(np.float32)
                        + new_weight * new_alpha.astype(np.float32)
                    ).clip(0, 255).astype(np.uint8)
                    abs_blend_frame = abs_start + i
                    mat_filename = os.path.join(core.matting_dir, f"{abs_blend_frame:05d}", f"{object_id}.png")
                    os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                    cv2.imwrite(mat_filename, blended)
            # -----------------------------------------

            # Collect the last `overlap` committed alphas (original resolution) for the
            # next batch's boundary blend, then write all committed frames to disk.
            current_boundary_alphas = []
            for i, frame_out in enumerate(committed_output):
                abs_frame = abs_start + output_start_offset + i
                alpha = cv2.cvtColor(frame_out, cv2.COLOR_RGB2GRAY)

                # Restore to the cropped region's pixel dimensions first, then expand
                # back into the full frame canvas so the saved matte is always full-size.
                if crop_rect is not None:
                    crop_w = cx2 - cx1 + 1
                    crop_h = cy2 - cy1 + 1
                    alpha = self._restore_image_size(alpha, (crop_w, crop_h))
                    final_alpha = core.expand_to_full(alpha, crop_rect, original_w, original_h)
                else:
                    final_alpha = self._restore_image_size(alpha, (original_w, original_h))

                if i >= len(committed_output) - overlap:
                    current_boundary_alphas.append(final_alpha.copy())
 
                mat_filename = os.path.join(core.matting_dir, f"{abs_frame:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, final_alpha)
 
                if abs_frame % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(abs_frame)
                        QApplication.processEvents()
                    except Exception:
                        pass

            prev_boundary_alphas = current_boundary_alphas if len(current_boundary_alphas) == overlap else None
 
            # Update progress
            pbar.update(1)
            batches_completed += 1
            current_progress = int(((operations_completed + batches_completed) * 100) / max(total_operations, 1))
            progress_dialog.setValue(min(current_progress, 99))
            QApplication.processEvents()
 
            core.DeviceManager.clear_cache()
 
        return batches_completed

# ---------------------------------------------------------------------------
# Manager Selection
# ---------------------------------------------------------------------------

def create_matting_manager() -> MattingManager:
    """
    Instantiate the correct matting manager based on the 'matting_model' session setting.
    """
    settings_mgr = get_settings_manager()
    matting_model = settings_mgr.get_session_setting("matany_model", "MatAnyone")
    if matting_model == "VideoMaMa":
        return VideoMaMaManager()
    return MatAnyManager()
