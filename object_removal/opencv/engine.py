"""OpenCV frame inpainting engine."""
import cv2
import os
import numpy as np
from tqdm import tqdm
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt
from sammie import core
from sammie.settings_manager import get_settings_manager
from object_removal.base import RemovalManager

class OpenCVRemovalManager(RemovalManager):
    def run(self, points, parent_window=None):
        return self.run_object_removal_cv(points, parent_window)

    def unload(self):
        pass

    def run_object_removal_cv(self, points_list, parent_window):
        """
        Run OpenCV object removal (inpainting) on all frames with points.
        Processes per frame instead of per object and combines masks for all objects.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        frame_count = core.VideoInfo.total_frames
        settings_mgr = get_settings_manager()

        # Get in/out points from settings
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)

        # Determine frame range to process
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1

        print(f"Processing removal from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get settings
        inpaint_method = settings_mgr.get_session_setting("inpaint_method", "Telea")
        inpaint_radius = settings_mgr.get_session_setting("inpaint_radius", 3)
        grow = settings_mgr.get_session_setting("grow", 0)  # segmentation grow
        inpaint_grow = settings_mgr.get_session_setting("inpaint_grow", 0)  # object removal grow
        inpaint_grow = inpaint_grow + grow
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        # Convert method string to OpenCV constant
        if inpaint_method == "Telea":
            cv2_method = cv2.INPAINT_TELEA
        elif inpaint_method == "Navier-Stokes":
            cv2_method = cv2.INPAINT_NS
        else:
            print(f"Unknown inpaint method: {inpaint_method}, defaulting to Telea")
            cv2_method = cv2.INPAINT_TELEA

        # Get unique object IDs
        object_ids = sorted(list(set(p['object_id'] for p in points_list if 'object_id' in p)))
        if not object_ids:
            print("No objects found for removal")
            return 0

        # Create progress dialog
        progress_dialog = QProgressDialog("Running object removal...", "Cancel", 0, frames_to_process, parent_window)
        progress_dialog.setWindowTitle("Object Removal Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()

        # Terminal progress bar
        tqdm_bar = tqdm(total=frame_count, desc="Object Removal", unit="frame", ncols=80)

        # Create output directory if it doesn't exist (don't clear existing frames)
        os.makedirs(core.removal_dir, exist_ok=True)

        extension = core.get_frame_extension()
        operations_completed = 0

        # Process each frame in the range
        for frame_number in range(start_frame, end_frame + 1):
            if progress_dialog.wasCanceled():
                break

            frame_filename = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")
            if not os.path.exists(frame_filename):
                operations_completed += 1
                tqdm_bar.update(1)
                progress_dialog.setValue(operations_completed)
                QApplication.processEvents()
                continue

            frame = cv2.imread(frame_filename)
            if frame is None:
                operations_completed += 1
                tqdm_bar.update(1)
                progress_dialog.setValue(operations_completed)
                QApplication.processEvents()
                continue

            # Combine masks from all objects on this frame
            combined_mask = np.zeros(frame.shape[:2], np.uint8)
            for object_id in object_ids:
                mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{object_id}.png")
                if os.path.exists(mask_filename):
                    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Skip if no mask present, copy original frame
            if not np.any(combined_mask):
                output_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, frame)
                operations_completed += 1
                tqdm_bar.update(1)
                progress_dialog.setValue(operations_completed)
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception:
                        pass
                    QApplication.processEvents()
                continue

            # Apply mask grow/shrink if requested
            if inpaint_grow != 0:
                combined_mask = core.grow_shrink(combined_mask, inpaint_grow)

            # Run inpainting
            try:
                result = cv2.inpaint(frame, combined_mask, inpaint_radius, cv2_method)
                output_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, result)

            except Exception as e:
                print(f"Error inpainting frame {frame_number}: {e}")
                output_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, frame)

            operations_completed += 1
            tqdm_bar.update(1)
            progress_dialog.setValue(operations_completed)

            # UI updates
            if frame_number % display_update_frequency == 0:
                progress_dialog.setValue(operations_completed)
                try:
                    parent_window.frame_slider.setValue(frame_number)
                except Exception as e:
                    print(f"Error updating display: {e}")
                QApplication.processEvents()

        # Finalize
        tqdm_bar.close()
        if progress_dialog.wasCanceled():
            print("Object removal cancelled — partial results kept.")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(frames_to_process)
            if frame_count == frames_to_process:  # only set propagated if entire video was processed
                self.propagated = True
            else:
                self.propagated = False
            print("Object removal completed successfully.")
            self._notify('removal_complete')
            return 1
