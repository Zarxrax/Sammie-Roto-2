# sammie/sammie.py
import cv2
import os
import numpy as np
import shutil
import torch
import re
import glob
import zipfile
import threading
import queue
import multiprocessing
from tqdm import tqdm
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication, QMessageBox
from sam2.build_sam import build_sam2_video_predictor
from sammie import core
from sammie.smooth import run_smoothing_model, prepare_smoothing_model
from sammie.duplicate_frame_handler import replace_similar_matte_frames
from sammie.settings_manager import get_settings_manager
from sammie.gui_widgets import show_message_dialog
from sammie.model_downloader import ensure_models

smoothing_model = None  # global variable needed to avoid complexity of passing the model around


# .........................................................................................
# SAM2 / EfficientTAM segmentation
# .........................................................................................

class SamManager:
    def __init__(self):
        self.model = None
        self.loaded_model_name = None
        self.predictor = None
        self.inference_state = None
        self.propagated = False  # whether we have propagated the masks
        self.deduplicated = False  # whether we have deduplicated the masks
        self.callbacks = []  # Add callbacks for segmentation events

    def add_callback(self, callback):
        """Add callback for segmentation events"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")

    def load_segmentation_model(self, model=None, parent_window=None):
        if model is None:
            settings_mgr = get_settings_manager()
            sam_model = settings_mgr.get_session_setting("sam_model", "Base")
        else:
            sam_model = model
        core.DeviceManager.clear_cache()
        device = core.DeviceManager.get_device()
        if sam_model == "Large":
            print("Loaded SAM2 Large model")
            checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"
        elif sam_model == "Base":
            print("Loaded SAM2 Base model")
            checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "./configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif sam_model == "Efficient":
            print("Loaded EfficientTAM 512x512 model")
            checkpoint = "./checkpoints/efficienttam_s_512x512.pt"
            model_cfg = "./configs/sam2.1/efficienttam_s_512x512.yaml"

        # Check if files exist
        if not ensure_models(sam_model, parent=parent_window):
            return False

        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
        self.loaded_model_name = sam_model
        return True  # model loaded successfully

    def unload_segmentation_model(self):
        """Unload the SAM model and clear cache"""
        self.predictor = None
        self.inference_state = None
        core.DeviceManager.clear_cache()
        print("Unloaded Segmentation model")

    def offload_model_to_cpu(self):
        """Offload SAM2 model to CPU to free VRAM"""
        device = core.DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do

        if self.predictor is not None:
            self.predictor.to('cpu')
            core.DeviceManager.clear_cache()

    def load_model_to_device(self):
        """Load SAM2 model back to the active device"""
        device = core.DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do

        if self.predictor is not None:
            self.predictor.to(device)

    def initialize_predictor(self):
        self.inference_state = self.predictor.init_state(
            video_path=core.frames_dir, async_loading_frames=True, offload_video_to_cpu=True
        )

    def segment_image(self, frame_number, object_id, input_points, input_labels):
        extension = core.get_frame_extension()
        frame_filename = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")
        if os.path.exists(frame_filename):
            #self.predictor.reset_state(self.inference_state)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_number,
                obj_id=object_id,
                points=input_points,
                labels=input_labels,
            )
            # Save the segmentation masks
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{out_obj_id}.png")
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                mask = (mask * 255).astype(np.uint8)
                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                cv2.imwrite(mask_filename, mask)

            # Notify that segmentation is complete
            self._notify('segmentation_complete', frame=frame_number, object_id=object_id, out_obj_ids=out_obj_ids)

    def preview_point(self, frame_number, object_id, all_points, preview_x, preview_y, is_positive):
        """Run a preview using the real video predictor, then revert the state."""
        if self.predictor is None or self.inference_state is None:
            return None
        try:
            existing = [p for p in all_points
                        if p['frame'] == frame_number and p['object_id'] == object_id]

            # Build preview point set
            preview_points = np.array([[p['x'], p['y']] for p in existing] + [[preview_x, preview_y]], dtype=np.float32)
            preview_labels = np.array([1 if p['positive'] else 0 for p in existing] + [1 if is_positive else 0], dtype=np.int32)

            # Step 1: run with preview point
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_number,
                obj_id=object_id,
                points=preview_points,
                labels=preview_labels,
                clear_old_points=True,
            )

            # Capture the preview mask
            preview_mask = None
            for i, oid in enumerate(out_obj_ids):
                if oid == object_id:
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    preview_mask = (mask * 255).astype(np.uint8)
                    break

            # Step 2: revert by replaying the original points
            if existing:
                orig_points = np.array([[p['x'], p['y']] for p in existing], dtype=np.float32)
                orig_labels = np.array([1 if p['positive'] else 0 for p in existing], dtype=np.int32)
                self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_number,
                    obj_id=object_id,
                    points=orig_points,
                    labels=orig_labels,
                    clear_old_points=True,
                )
            else:
                self.predictor.reset_state(self.inference_state)

            return preview_mask

        except Exception as e:
            print(f"Preview error: {e}")
            return None

    def replay_points(self, points_list):
        """Replay all points incrementally to rebuild masks."""
        frame_count = core.VideoInfo.total_frames
        self.predictor.reset_state(self.inference_state)

        for frame_number in range(frame_count):
            frame_points = [p for p in points_list if p['frame'] == frame_number]
            if not frame_points:
                continue

            frame_object_ids = {p['object_id'] for p in frame_points}
            for object_id in frame_object_ids:
                filtered_points = [
                    (p['x'], p['y'], p['positive'])
                    for p in frame_points if p['object_id'] == object_id
                ]
                for i in range(1, len(filtered_points) + 1):
                    subset = filtered_points[:i]
                    input_points = np.array([(x, y) for x, y, _ in subset], dtype=np.float32)
                    input_labels = np.array([1 if pos else 0 for _, _, pos in subset], dtype=np.int32)
                    try:
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=frame_number,
                            obj_id=object_id,
                            points=input_points,
                            labels=input_labels,
                            clear_old_points=True
                        )
                    except Exception as e:
                        print(f"Error during prediction for frame {frame_number}, object {object_id}, point {i}: {e}")
                        continue

                # Save masks only after the final point for this object
                for j, out_obj_id in enumerate(out_obj_ids):
                    mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{out_obj_id}.png")
                    mask = (out_mask_logits[j] > 0.0).cpu().numpy().squeeze()
                    mask = (mask * 255).astype(np.uint8)
                    try:
                        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                        cv2.imwrite(mask_filename, mask)
                    except Exception as e:
                        print(f"Error saving mask for frame {frame_number}, object {out_obj_id}: {e}")

        self._notify('replay_complete')

    def track_objects(self, parent_window):
        frame_count = core.VideoInfo.total_frames
        progress_dialog = QProgressDialog("Tracking...", "Cancel", 0, 100, parent_window)
        progress_dialog.setWindowTitle("Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()

        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        # Figure out the frame range to track
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        if in_point is None:
            in_point = 0
        frames_to_track = None
        total_frames = frame_count
        if out_point is not None:
            frames_to_track = out_point - in_point
        if frames_to_track is not None:
            total_frames = frames_to_track + 1

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state, start_frame_idx=in_point, max_frame_num_to_track=frames_to_track):
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_filename = os.path.join(core.mask_dir, f"{out_frame_idx:05d}", f"{out_obj_id}.png")
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                mask = (mask * 255).astype(np.uint8)
                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                cv2.imwrite(mask_filename, mask)

            # Update progress dialog
            frames_processed = out_frame_idx - in_point + 1
            progress_dialog.setValue(int(frames_processed * 100 / total_frames))

            # Update display at the specified frequency
            if out_frame_idx % display_update_frequency == 0:
                try:
                    parent_window.frame_slider.setValue(out_frame_idx)
                except Exception as e:
                    print(f"Error updating display: {e}")

            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                break

        if not progress_dialog.wasCanceled():
            progress_dialog.setValue(100)
            if total_frames == frame_count:
                self.propagated = True
            else:
                self.propagated = False
            print("Tracking completed")
            return 1
        else:
            progress_dialog.close()
            self.propagated = False
            print("Tracking cancelled")
            return 0

    def clear_tracking(self):
        """Clear tracking data by deleting all masks, this needs to be followed up by replay_points"""
        if os.path.exists(core.mask_dir):
            shutil.rmtree(core.mask_dir)
        os.makedirs(core.mask_dir)
        self.predictor.reset_state(self.inference_state)
        core.DeviceManager.clear_cache()
        if self.propagated:
            print("Tracking data cleared")
        self.propagated = False
        self.deduplicated = False


# .........................................................................................
# Smoothing model
# .........................................................................................

def load_smoothing_model():
    global smoothing_model
    if smoothing_model is None:
        device = core.DeviceManager.get_device()
        try:
            smoothing_model = prepare_smoothing_model("./checkpoints/1x_binary_mask_smooth.pth", device)
        except Exception as e:
            print(f"Warning: Could not load antialiasing model: {e}")
            smoothing_model = None


# .........................................................................................
# View / display handlers
# .........................................................................................

def update_image(slider_value, view_options, points, return_numpy=False, object_id_filter=None, preview_mask=None):
    """Main image update function - delegates to specific view handlers

    Args:
        slider_value: Frame number
        view_options: Dictionary of view options
        points: List of point dictionaries
        return_numpy: If True, return numpy array; if False, return QPixmap
        object_id_filter: If specified, only process masks for this object ID

    Returns:
        QPixmap or numpy array depending on return_numpy parameter
    """
    view_mode = view_options.get("view_mode", "Segmentation-Edit")

    if view_mode == "Segmentation-Edit":
        return _handle_segmentation_edit_view(slider_value, view_options, points, return_numpy, object_id_filter, preview_mask)
    elif view_mode == "Segmentation-Matte":
        return _handle_segmentation_matte_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-BGcolor":
        return _handle_segmentation_bgcolor_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-Alpha":
        return _handle_segmentation_alpha_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-Matte":
        return _handle_matting_matte_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-BGcolor":
        return _handle_matting_bgcolor_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-Alpha":
        return _handle_matting_alpha_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "ObjectRemoval":
        return _handle_object_removal_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "None":
        return _handle_none_view(slider_value, return_numpy)
    else:
        print(f"Unknown view mode: {view_mode}")
        return None


def load_removal_frame(frame_number):
    """
    Load the object removal frame image from disk.
    If the frame does not exist, load the base frame instead.
    """
    frame_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
    if os.path.exists(frame_filename):
        image = cv2.imread(frame_filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return core.load_base_frame(frame_number)


def _convert_to_qpixmap(image):
    """Convert NumPy array to QPixmap"""
    if image is None:
        return QPixmap.fromImage(QImage())
    image = image.copy()
    height, width = image.shape[:2]

    if len(image.shape) == 2:  # Grayscale
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif image.shape[2] == 3:  # RGB
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
    else:  # RGBA
        bytes_per_line = 4 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)

    return QPixmap.fromImage(q_image)


def _handle_none_view(frame_number, return_numpy=False):
    """Handle None view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None
    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_edit_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None, preview_mask=None):
    """Handle Segmentation-Edit view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    image = apply_postprocessing_to_display(image, frame_number, points, view_options, object_id_filter, preview_mask)

    highlighted_points = view_options.get('highlighted_point', None)

    if highlighted_points is None:
        highlighted_points = None
    elif isinstance(highlighted_points, list):
        highlighted_points = highlighted_points.copy()
    else:
        highlighted_points = [highlighted_points]

    image = draw_points(image, frame_number, points, highlighted_points)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_matte_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Matte view"""
    mask = core.load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        return None

    mask = core.apply_mask_postprocessing(mask)
    mask_3channel = np.stack([mask] * 3, axis=-1)

    if view_options.get("antialias", True):
        global smoothing_model
        if smoothing_model is None:
            load_smoothing_model()
        if smoothing_model is not None:
            device = core.DeviceManager.get_device()
            mask_3channel = run_smoothing_model(mask_3channel, smoothing_model, device)

    if return_numpy:
        return mask_3channel
    else:
        return _convert_to_qpixmap(mask_3channel)


def _handle_segmentation_bgcolor_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-BGcolor view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = core.load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    mask = core.apply_mask_postprocessing(mask)
    mask_3channel = np.stack([mask] * 3, axis=-1)

    if view_options.get("antialias", True):
        global smoothing_model
        if smoothing_model is None:
            load_smoothing_model()
        if smoothing_model is not None:
            device = core.DeviceManager.get_device()
            mask_3channel = run_smoothing_model(mask_3channel, smoothing_model, device)

    bgcolor = view_options.get("bgcolor", (0, 255, 0))
    bgcolor_bgr = np.array(bgcolor)
    alpha_channel = mask_3channel / 255.0
    image = (image * alpha_channel) + (bgcolor_bgr * (1 - alpha_channel))
    image = image.astype(np.uint8)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_segmentation_alpha_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Alpha view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    image_rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    image_rgba[:, :, :3] = image
    image_rgba[:, :, 3] = 255

    mask = core.load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        return _convert_to_qpixmap(image_rgba) if not return_numpy else image_rgba

    mask = core.apply_mask_postprocessing(mask)

    if view_options.get("antialias", True):
        global smoothing_model
        if smoothing_model is None:
            load_smoothing_model()
        if smoothing_model is not None:
            device = core.DeviceManager.get_device()
            mask_3channel = np.stack([mask] * 3, axis=-1)
            mask_3channel = run_smoothing_model(mask_3channel, smoothing_model, device)
            mask = mask_3channel[:, :, 0]

    image_rgba[:, :, 3] = mask

    if return_numpy:
        return image_rgba
    else:
        return _convert_to_qpixmap(image_rgba)


def _handle_matting_matte_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-Matte view"""
    mask = core.load_masks_for_frame(frame_number, points, return_combined=True,
                                object_id_filter=object_id_filter, folder=core.matting_dir)
    if mask is None:
        return None

    mask = core.apply_matany_postprocessing(mask)
    mask_3channel = np.stack([mask] * 3, axis=-1)

    if return_numpy:
        return mask_3channel
    else:
        return _convert_to_qpixmap(mask_3channel)


def _handle_matting_bgcolor_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-BGcolor view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    mask = core.load_masks_for_frame(frame_number, points, return_combined=True,
                                object_id_filter=object_id_filter, folder=core.matting_dir)
    if mask is None:
        return _convert_to_qpixmap(image) if not return_numpy else image

    mask = core.apply_matany_postprocessing(mask)
    mask_3channel = np.stack([mask] * 3, axis=-1)

    bgcolor = view_options.get("bgcolor", (0, 255, 0))
    bgcolor_bgr = np.array(bgcolor)
    alpha_channel = mask_3channel / 255.0
    image = (image * alpha_channel) + (bgcolor_bgr * (1 - alpha_channel))
    image = image.astype(np.uint8)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def _handle_matting_alpha_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-Alpha view"""
    image = core.load_base_frame(frame_number)
    if image is None:
        return None

    image_rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    image_rgba[:, :, :3] = image
    image_rgba[:, :, 3] = 255

    mask = core.load_masks_for_frame(frame_number, points, return_combined=True,
                                object_id_filter=object_id_filter, folder=core.matting_dir)
    if mask is None:
        return _convert_to_qpixmap(image_rgba) if not return_numpy else image_rgba

    mask = core.apply_matany_postprocessing(mask)
    image_rgba[:, :, 3] = mask

    if return_numpy:
        return image_rgba
    else:
        return _convert_to_qpixmap(image_rgba)


def _handle_object_removal_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Object Removal view"""
    image = load_removal_frame(frame_number)
    if image is None:
        return None

    mask = core.load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        return None

    mask = core.apply_mask_postprocessing(mask)

    settings_mgr = get_settings_manager()
    grow = settings_mgr.get_session_setting("inpaint_grow", 0)
    mask = core.grow_shrink(mask, grow)

    if view_options.get("show_removal_mask", True):
        image = draw_removal_overlay(image, mask)

    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)


def draw_masks(image, processed_masks):
    """Draw masks on the current frame (expects preprocessed masks)"""
    if not processed_masks:
        return image

    combined_colored_mask = np.zeros_like(image, dtype=np.uint8)
    mask_binary = np.zeros(image.shape[:2], dtype=bool)

    for object_id, mask in processed_masks.items():
        color = np.array(core.PALETTE[object_id % len(core.PALETTE)], dtype=np.uint8)
        mask_bin = mask > 0
        mask_binary |= mask_bin
        combined_colored_mask[mask_bin] = color

    if np.any(mask_binary):
        overlay = image.copy()
        overlay[mask_binary] = cv2.addWeighted(
            image[mask_binary], 0.5,
            combined_colored_mask[mask_binary], 0.5, 0
        )
        return overlay
    else:
        return image


def draw_removal_overlay(image, mask):
    """Draw masked overlay on the current frame for object removal"""
    mask_norm = mask.astype(np.float32) / 255.0
    mask_3ch = np.dstack([mask_norm] * 3)
    color_layer = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    overlay = np.where(
        (mask_3ch > 0),
        cv2.addWeighted(image, 1 - 0.5, color_layer, 0.5, 0),
        image
    )
    return overlay


def draw_contours(image, processed_masks):
    """Draw colored contours on the current frame (expects preprocessed masks)"""
    if not processed_masks:
        return image

    overlay = image.copy()
    kernel = np.ones((3, 3), np.uint8)

    for object_id, mask in processed_masks.items():
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        border_color = core.PALETTE[object_id % len(core.PALETTE)]
        overlay[edges > 0] = border_color

    return overlay


def draw_points(image, frame_number, points, highlighted_points=None):
    """Draw points on image, with optional highlighting"""
    frame_points = [p for p in points if p['frame'] == frame_number]
    if not frame_points:
        return image

    is_highlighted = False

    for point in frame_points:
        is_highlighted = False
        if highlighted_points:
            for point_in_highlighted_list in highlighted_points:
                is_highlighted = (
                    point_in_highlighted_list['frame'] == point['frame'] and
                    point_in_highlighted_list['x'] == point['x'] and
                    point_in_highlighted_list['y'] == point['y']
                )
                if is_highlighted:
                    highlighted_points.remove(point_in_highlighted_list)
                    break

        if is_highlighted:
            center = (point['x'], point['y'])
            cv2.circle(image, center, 9, (0, 128, 255), 3)
            point_color = (0, 255, 0) if point['positive'] else (255, 0, 0)
            cv2.circle(image, center, 5, (255, 255, 0), 2)
            cv2.circle(image, center, 4, point_color, -1)
        else:
            point_color = (0, 255, 0) if point['positive'] else (255, 0, 0)
            center = (point['x'], point['y'])
            cv2.circle(image, center, 5, (255, 255, 0), 2)
            cv2.circle(image, center, 4, point_color, -1)

    return image


def apply_postprocessing_to_display(image, frame_number, points, view_options, object_id_filter=None, preview_mask=None):
    """Apply postprocessing to masks and draw them on the image for display"""
    raw_masks = core.load_masks_for_frame(
        frame_number, points, return_combined=False, object_id_filter=object_id_filter
    )

    if raw_masks:
        processed_masks = {
            object_id: core.apply_mask_postprocessing(mask) for object_id, mask in raw_masks.items()
        }
    else:
        processed_masks = {}

    # Substitute the preview mask for the selected object if provided
    if preview_mask is not None and object_id_filter is not None:
        processed_masks[object_id_filter] = core.apply_mask_postprocessing(preview_mask)

    if view_options.get("show_masks", True):
        image = draw_masks(image, processed_masks)
    if view_options.get("show_outlines", True):
        image = draw_contours(image, processed_masks)

    return image


# .........................................................................................
# Mask / session utilities
# .........................................................................................

def deduplicate_masks(parent_window):
    """Deduplicate similar masks using settings threshold"""
    settings_mgr = get_settings_manager()
    threshold = settings_mgr.app_settings.dedupe_threshold
    return replace_similar_matte_frames(parent_window, threshold)


def remove_backup_mattes():
    if os.path.exists(core.backup_dir):
        shutil.rmtree(core.backup_dir)


# .........................................................................................
# Video / image I/O
# .........................................................................................

def load_video(video_file, parent_window):
    """Load video and save frames as images using multi-threaded writers"""
    if os.path.exists(core.temp_dir):
        shutil.rmtree(core.temp_dir)
    os.makedirs(core.frames_dir)
    os.makedirs(core.mask_dir)
    os.makedirs(core.matting_dir)
    print(f"Loading video: {video_file}")

    progress_dialog = QProgressDialog("Loading video...", "Cancel", 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()

    cap = cv2.VideoCapture(video_file)
    core.VideoInfo.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    core.VideoInfo.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    core.VideoInfo.fps = cap.get(cv2.CAP_PROP_FPS)
    core.VideoInfo.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = core.VideoInfo.total_frames

    frame_count = 0
    settings_mgr = get_settings_manager()
    frame_format = settings_mgr.get_app_setting("frame_format", "png")

    # --- Threaded frame writing setup ---
    save_q = queue.Queue(maxsize=100)
    num_workers = max(2, multiprocessing.cpu_count() // 2)

    def save_worker():
        while True:
            item = save_q.get()
            if item is None:
                save_q.task_done()
                break
            path, frame = item
            try:
                cv2.imwrite(path, frame)
            except Exception as e:
                print(f"Error writing {path}: {e}")
            save_q.task_done()

    writers = []
    for _ in range(num_workers):
        t = threading.Thread(target=save_worker, daemon=True)
        t.start()
        writers.append(t)

    with tqdm(total=total_frames) as progress:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(core.frames_dir, f"{frame_count:05d}.{frame_format}")
            save_q.put((frame_filename, frame))
            frame_count += 1
            progress.update(1)
            progress_dialog.setValue(frame_count * 100 / total_frames)
            QApplication.processEvents()

            if progress_dialog.wasCanceled():
                cap.release()
                while not save_q.empty():
                    try:
                        save_q.get_nowait()
                    except queue.Empty:
                        pass
                for _ in writers:
                    save_q.put(None)
                for t in writers:
                    t.join()
                if os.path.exists(core.temp_dir):
                    shutil.rmtree(core.temp_dir)
                progress_dialog.close()
                print("Operation cancelled by user.")
                return 0

        cap.release()

    save_q.join()
    for _ in writers:
        save_q.put(None)
    for t in writers:
        t.join()

    progress_dialog.setValue(100)
    progress_dialog.close()

    core.VideoInfo.total_frames = frame_count
    return frame_count



def detect_image_sequence(image_path):
    """
    Detect if an image is part of a sequence based on common naming patterns.
    Returns (is_sequence, sequence_files) or (False, [])
    """
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)

    patterns = [
        r'^(.+?)(\d{4,})$',
        r'^(.+?)(\d{3})$',
        r'^(.+?)(\d{2})$',
        r'^(.+?)_(\d+)$',
        r'^(.+?)\-(\d+)$',
        r'^(.+?)\.(\d+)$',
    ]

    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            base_name = match.group(1)

            if pattern == r'^(.+?)(\d{4,})$':
                glob_pattern = f"{base_name}*{ext}"
            elif pattern == r'^(.+?)(\d{3})$':
                glob_pattern = f"{base_name}???{ext}"
            elif pattern == r'^(.+?)(\d{2})$':
                glob_pattern = f"{base_name}??{ext}"
            else:
                separator = pattern.split('(\\d+)')[0][-1]
                glob_pattern = f"{base_name}{separator}*{ext}"

            search_path = os.path.join(directory, glob_pattern)
            potential_files = glob.glob(search_path)

            sequence_files = []
            for file_path in potential_files:
                file_name = os.path.basename(file_path)
                file_base = os.path.splitext(file_name)[0]
                if re.match(pattern, file_base):
                    sequence_files.append(file_path)

            def natural_sort_key(path):
                base_name = os.path.splitext(os.path.basename(path))[0]
                match = re.match(pattern, base_name)
                if match:
                    return (match.group(1), int(match.group(2)))
                return (base_name, 0)

            sequence_files.sort(key=natural_sort_key)

            if len(sequence_files) > 1:
                return True, sequence_files

    return False, []


def load_image_sequence(image_path, parent_window):
    """
    Load an image or image sequence. Detects sequences automatically and prompts user.
    """
    is_sequence, sequence_files = detect_image_sequence(image_path)
    files_to_load = [image_path]

    if is_sequence:
        msg_box = QMessageBox(parent_window)
        msg_box.setWindowTitle("Image Sequence Detected")
        msg_box.setText(f"The selected image appears to be part of a sequence with {len(sequence_files)} images.")
        msg_box.setInformativeText("Would you like to load the entire sequence or just the single image?")

        sequence_button = msg_box.addButton("Load Sequence", QMessageBox.AcceptRole)
        single_button = msg_box.addButton("Load Single Image", QMessageBox.RejectRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

        msg_box.exec()

        if msg_box.clickedButton() == sequence_button:
            files_to_load = sequence_files
        elif msg_box.clickedButton() == single_button:
            files_to_load = [image_path]
        else:
            return 0

    if os.path.exists(core.temp_dir):
        shutil.rmtree(core.temp_dir)
    os.makedirs(core.frames_dir)
    os.makedirs(core.mask_dir)
    os.makedirs(core.matting_dir)

    print(f"Loading {'image sequence' if len(files_to_load) > 1 else 'image'}: {len(files_to_load)} file(s)")

    progress_dialog = QProgressDialog("Loading images...", "Cancel", 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()

    settings_mgr = get_settings_manager()
    app_frame_format = settings_mgr.get_app_setting("frame_format", "png")

    first_image = cv2.imread(files_to_load[0])
    if first_image is None:
        progress_dialog.close()
        show_message_dialog(parent_window, title="Error",
                            message=f"Could not load image: {files_to_load[0]}", type="critical")
        return 0

    core.VideoInfo.height, core.VideoInfo.width = first_image.shape[:2]
    core.VideoInfo.fps = 24.0
    core.VideoInfo.total_frames = len(files_to_load)

    for frame_count, source_path in enumerate(files_to_load):
        image = cv2.imread(source_path)
        if image is None:
            print(f"Warning: Could not load {source_path}, skipping...")
            continue

        source_ext = os.path.splitext(source_path)[1].lower()
        if source_ext in ['.png', '.jpg', '.jpeg']:
            output_ext = source_ext.lstrip('.')
            frame_filename = os.path.join(core.frames_dir, f"{frame_count:05d}.{output_ext}")
            shutil.copy2(source_path, frame_filename)
        else:
            frame_filename = os.path.join(core.frames_dir, f"{frame_count:05d}.{app_frame_format}")
            cv2.imwrite(frame_filename, image)

        progress_dialog.setValue((frame_count + 1) * 100 // len(files_to_load))
        QApplication.processEvents()

        if progress_dialog.wasCanceled():
            if os.path.exists(core.temp_dir):
                shutil.rmtree(core.temp_dir)
            progress_dialog.close()
            return 0

    progress_dialog.setValue(100)
    return core.VideoInfo.total_frames


def resume_session():
    if os.path.exists(core.temp_dir):
        if os.path.exists(core.frames_dir) and os.listdir(core.frames_dir):
            print("Resuming previous session...")
            restore_video_info()
            return core.VideoInfo.total_frames


def restore_video_info():
    if not os.path.exists(core.frames_dir):
        return 0
    image = core.load_base_frame(0)
    if image is not None:
        height, width, channels = image.shape
        core.VideoInfo.width = width
        core.VideoInfo.height = height
        core.VideoInfo.fps = 24
        extension = core.get_frame_extension()
        core.VideoInfo.total_frames = len([f for f in os.listdir(core.frames_dir) if f.endswith(f".{extension}")])

        settings_mgr = get_settings_manager()
        if settings_mgr.session_exists():
            session_width = settings_mgr.get_session_setting("video_width", 0)
            session_height = settings_mgr.get_session_setting("video_height", 0)
            session_fps = settings_mgr.get_session_setting("video_fps", 0)
            session_frames = settings_mgr.get_session_setting("total_frames", 0)

            if session_width > 0 and session_height > 0:
                core.VideoInfo.width = session_width
                core.VideoInfo.height = session_height
            if session_fps > 0:
                core.VideoInfo.fps = session_fps
            if session_frames > 0:
                core.VideoInfo.total_frames = session_frames


def load_project(file_name, parent_window):
    if os.path.exists(core.temp_dir):
        shutil.rmtree(core.temp_dir)
    os.makedirs(core.temp_dir, exist_ok=True)
    progress = None

    try:
        with zipfile.ZipFile(file_name, 'r') as zipf:
            file_list = zipf.namelist()
            total_files = len(file_list)

        if total_files == 0:
            return True

        progress = QProgressDialog("Extracting files...", "", 0, total_files, parent_window)
        progress.setWindowTitle("Extracting Backup")
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()

        with zipfile.ZipFile(file_name, 'r') as zipf:
            for i, file_name in enumerate(file_list):
                progress.setValue(i)
                progress.setLabelText(f"Extracting: {os.path.basename(file_name)}")
                QApplication.processEvents()
                zipf.extract(file_name, core.temp_dir)

        progress.setValue(total_files)
        progress.setLabelText("Extraction completed!")
        QApplication.processEvents()
        return True

    except Exception as e:
        if progress:
            progress.close()
        raise e

    finally:
        if progress:
            progress.close()


def save_project(file_name, parent_window):
    total_files = 0
    all_files = []

    for root, dirs, files in os.walk(core.temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, core.temp_dir)
            all_files.append((file_path, arcname))
            total_files += 1

    progress = QProgressDialog("Backing up files...", "Cancel", 0, total_files, parent_window)
    progress.setWindowTitle("Creating Backup")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.show()
    QApplication.processEvents()

    try:
        with zipfile.ZipFile(file_name, 'w', zipfile.ZIP_STORED) as zipf:
            for i, (file_path, arcname) in enumerate(all_files):
                if progress.wasCanceled():
                    try:
                        os.remove(file_name)
                    except Exception:
                        pass
                    return 0

                progress.setValue(i)
                progress.setLabelText(f"Adding: {os.path.basename(file_path)}")
                QApplication.processEvents()
                zipf.write(file_path, arcname)

        progress.setValue(total_files)
        QApplication.processEvents()

    except Exception as e:
        progress.close()
        try:
            os.remove(file_name)
        except Exception:
            pass
        raise e

    finally:
        if not progress.wasCanceled():
            progress.close()

    return 1
