from sammie import image_ops
import os
import numpy as np
import torch
import gc
from tqdm import tqdm
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt
from sammie import core
from sammie.settings_manager import get_settings_manager

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
            return image_ops.resize(image, (new_w, new_h), interpolation=image_ops.INTER_AREA)
    
    def _restore_image_size(self, image, original_size):
        """Restore image to original size. original_size must be (w, h) as expected by image_ops."""
        original_w, original_h = original_size
        restored_image = image_ops.resize(image, (original_w, original_h), interpolation=image_ops.INTER_LINEAR)
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
                m = image_ops.imread(mask_filename, image_ops.IMREAD_GRAYSCALE)
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

        mask = image_ops.imread(mask_filename, image_ops.IMREAD_GRAYSCALE)
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

