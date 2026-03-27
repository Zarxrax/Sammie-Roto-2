# sammie/core.py
import cv2
import os
import numpy as np
import torch
import warnings
from sammie.settings_manager import get_settings_manager

# .........................................................................................
# Global variables
# .........................................................................................

temp_dir = "temp"
frames_dir = os.path.join(temp_dir, "frames")
mask_dir = os.path.join(temp_dir, "masks")
backup_dir = os.path.join(temp_dir, "masks_backup")
matting_dir = os.path.join(temp_dir, "matting")
removal_dir = os.path.join(temp_dir, "removal")

PALETTE = [
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
    (128, 128, 128), (64, 0, 0), (191, 0, 0), (64, 128, 0), (191, 128, 0), (64, 0, 128),
    (191, 0, 128), (64, 128, 128), (191, 128, 128), (0, 64, 0), (128, 64, 0), (0, 191, 0),
    (128, 191, 0), (0, 64, 128), (128, 64, 128)
]


class VideoInfo:
    width = 0
    height = 0
    fps = 0
    total_frames = 0


class DeviceManager:
    _device = None

    @classmethod
    def setup_device(cls):
        """Detect and set up the best available device"""
        if cls._device is not None:
            return cls._device  # already set

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("PyTorch version:", torch.__version__)

        settings_mgr = get_settings_manager()
        force_cpu = settings_mgr.get_app_setting("force_cpu", 0)

        if torch.cuda.is_available():
            cls._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            cls._device = torch.device("mps")
        elif torch.xpu.is_available():
            cls._device = torch.device("xpu")
        else:
            cls._device = torch.device("cpu")

        if force_cpu:
            cls._device = torch.device("cpu")

        print(f"Using device: {cls._device}")

        if cls._device.type == "cuda":
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                print("CUDA Compute Capability: ", torch.cuda.get_device_capability())
                # Enable bfloat16 for Ampere and newer
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    torch.autocast("cuda", dtype=torch.float16).__enter__()

        elif cls._device.type == "mps":
            torch.autocast("mps", dtype=torch.float16).__enter__()
        
        elif cls._device.type == "xpu":
            torch.autocast("xpu", dtype=torch.bfloat16).__enter__()

        return cls._device

    @classmethod
    def get_device(cls):
        """Return the already initialized device (or setup if needed)"""
        if cls._device is None:
            return cls.setup_device()
        return cls._device

    @classmethod
    def clear_cache(cls):
        if cls._device is None:
            return
        if cls._device.type == "cuda":
            torch.cuda.empty_cache()
        elif cls._device.type == "mps":
            torch.mps.empty_cache()
        elif cls._device.type == "xpu":
            torch.xpu.empty_cache()


class PointManager:
    def __init__(self):
        self.points = []  # List of dicts: {'frame': int, 'object_id': int, 'positive': bool, 'x': int, 'y': int}
        self.callbacks = []  # Callbacks for when points change

    def add_callback(self, callback):
        """Add callback for point changes"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except Exception as e:
                print(f"Point callback error: {e}")

    def add_point(self, frame, object_id, positive, x, y):
        """Add a point"""
        point = {'frame': frame, 'object_id': object_id, 'positive': positive, 'x': x, 'y': y}
        self.points.append(point)
        self._notify('add', point=point)
        settings_mgr = get_settings_manager()
        settings_mgr.save_points(self.points)
        return point

    def remove_point(self, frame, object_id, x, y):
        """Remove a specific point"""
        before_count = len(self.points)
        point_to_remove = None

        # Find the matching point
        for i, point in enumerate(self.points):
            if (point['frame'] == frame and
                point['object_id'] == object_id and
                point['x'] == x and
                point['y'] == y):
                point_to_remove = self.points.pop(i)
                break

        if point_to_remove:
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('remove_point', point=point_to_remove)
            return point_to_remove
        return None

    def remove_last(self):
        """Remove last point"""
        if self.points:
            point = self.points.pop()
            mask_filename = os.path.join(mask_dir, f'{point["frame"]:05d}', f'{point["object_id"]}.png')
            if os.path.exists(mask_filename):
                os.remove(mask_filename)
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('remove_last', point=point)
            return point
        return None

    def clear_all(self):
        """Clear all points"""
        if self.points:  # Only notify if there were points to clear
            self.points.clear()
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('clear_all')

    def clear_frame(self, frame):
        """Clear points for a frame"""
        import shutil
        before_count = len(self.points)
        points_to_remove = [p for p in self.points if p['frame'] == frame]
        self.points = [p for p in self.points if p['frame'] != frame]
        removed_count = before_count - len(self.points)

        if removed_count > 0:
            # Remove mask files for this frame
            frame_mask_dir = os.path.join(mask_dir, f"{frame:05d}")
            if os.path.exists(frame_mask_dir):
                shutil.rmtree(frame_mask_dir)
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('clear_frame', frame=frame, count=removed_count, points=points_to_remove)
        return removed_count

    def clear_object(self, object_id):
        """Clear points for an object"""
        before_count = len(self.points)
        points_to_remove = [p for p in self.points if p['object_id'] == object_id]
        self.points = [p for p in self.points if p['object_id'] != object_id]
        removed_count = before_count - len(self.points)

        if removed_count > 0:
            # Remove mask files for this object across all frames
            for point in points_to_remove:
                mask_filename = os.path.join(mask_dir, f'{point["frame"]:05d}', f'{object_id}.png')
                matting_filename = os.path.join(matting_dir, f'{point["frame"]:05d}', f'{object_id}.png')
                if os.path.exists(mask_filename):
                    os.remove(mask_filename)
                if os.path.exists(matting_filename):
                    os.remove(matting_filename)
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('clear_object', object_id=object_id, count=removed_count, points=points_to_remove)
        return removed_count

    def get_sam2_points(self, frame, object_id=None):
        """Get points in SAM2 format: (coordinates, labels)"""
        frame_points = [p for p in self.points if p['frame'] == frame]
        if object_id is not None:
            frame_points = [p for p in frame_points if p['object_id'] == object_id]

        if not frame_points:
            return [], []

        coordinates = [[p['x'], p['y']] for p in frame_points]
        labels = [1 if p['positive'] else 0 for p in frame_points]
        return coordinates, labels

    def get_points_for_frame(self, frame):
        """Get all points for a frame"""
        return [p for p in self.points if p['frame'] == frame]

    def get_all_points(self):
        """Get all points"""
        return self.points.copy()


# .........................................................................................
# Frame / mask loading utilities
# .........................................................................................

def get_frame_extension():
    """Get the frame file extension from session settings, fallback to PNG"""
    settings_mgr = get_settings_manager()
    frame_format = settings_mgr.get_session_setting("frame_format", "png")
    return frame_format


def load_base_frame(frame_number):
    """Load the base frame image from disk"""
    extension = get_frame_extension()
    frame_filename = os.path.join(frames_dir, f"{frame_number:05d}.{extension}")
    if os.path.exists(frame_filename):
        image = cv2.imread(frame_filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        print(f"{frame_filename} not found")
        return None


def load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=None, folder=None):
    """
    Load masks for a frame, returning either individual masks or a combined mask.

    Args:
        frame_number (int): Frame number to load masks for
        points (list): List of point dictionaries containing object_id information
        return_combined (bool): If True, return single combined mask. If False, return dict of individual masks.
        object_id_filter (int): only load masks for a specific object id
        folder: which mask folder to get images from; defaults to mask_dir

    Returns:
        If return_combined=True: Single numpy array (grayscale) or None if no masks
        If return_combined=False: Dict {object_id: mask_array} or empty dict if no masks
    """
    if folder is None:
        folder = mask_dir

    # Get unique object IDs from points
    object_ids = list(set(p['object_id'] for p in points if 'object_id' in p))

    # Filter by specific object ID if requested
    if object_id_filter is not None:
        object_ids = [obj_id for obj_id in object_ids if obj_id == object_id_filter]

    if not object_ids:
        return None if return_combined else {}

    individual_masks = {}

    # Load each mask file
    for object_id in object_ids:
        mask_filename = os.path.join(folder, f"{frame_number:05d}", f"{object_id}.png")
        if os.path.exists(mask_filename):
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                individual_masks[object_id] = mask
        else:
            # if mask doesn't exist, create a blank frame
            individual_masks[object_id] = np.zeros((VideoInfo.height, VideoInfo.width), dtype=np.uint8)

    if not individual_masks:
        return None if return_combined else {}

    if return_combined:
        # Combine all masks into a single mask (union operation)
        combined_mask = np.zeros((VideoInfo.height, VideoInfo.width), dtype=np.uint8)
        for mask in individual_masks.values():
            combined_mask = np.maximum(combined_mask, mask)
        return combined_mask
    else:
        return individual_masks


# .........................................................................................
# Mask postprocessing utilities
# .........................................................................................

def apply_mask_postprocessing(mask):
    """Apply postprocessing to a mask using current session settings"""
    settings_mgr = get_settings_manager()

    holes = settings_mgr.get_session_setting("holes", 0)
    dots = settings_mgr.get_session_setting("dots", 0)
    border_fix = settings_mgr.get_session_setting("border_fix", 0)
    grow = settings_mgr.get_session_setting("grow", 0)

    if holes > 0:
        mask = fill_small_holes(mask, holes)
    if dots > 0:
        mask = remove_small_dots(mask, dots)
    if border_fix > 0:
        mask = apply_border_fix(mask, border_fix)
    if grow != 0:
        mask = grow_shrink(mask, grow)

    return mask


def apply_matany_postprocessing(mask):
    """Apply postprocessing to MatAnyone results using current session settings"""
    settings_mgr = get_settings_manager()

    grow = settings_mgr.get_session_setting("matany_grow", 0)
    gamma = settings_mgr.get_session_setting("matany_gamma", 1.0)

    if grow != 0:
        mask = grow_shrink(mask, grow)
    if gamma != 1.0:
        mask = change_gamma(mask, gamma)

    return mask


def fill_small_holes(mask, holes_value):
    max_hole_area = holes_value ** 2
    filled_mask = mask.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area <= max_hole_area and hierarchy[0][i][3] != -1:  # Check if it's a hole (child contour)
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filled_mask


def remove_small_dots(mask, dots_value):
    max_dot_area = dots_value ** 2
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned_mask = np.zeros_like(mask)
    for label in range(1, num_labels):  # skip background
        if stats[label, cv2.CC_STAT_AREA] > max_dot_area:
            cleaned_mask[labels == label] = 255

    return cleaned_mask


def grow_shrink(mask, grow_value):
    kernel = np.ones((abs(grow_value) + 1, abs(grow_value) + 1), np.uint8)
    if grow_value > 0:
        return cv2.dilate(mask, kernel, iterations=1)
    elif grow_value < 0:
        return cv2.erode(mask, kernel, iterations=1)
    else:
        return mask


def apply_border_fix(mask, border_size):
    if border_size == 0:
        return mask
    height, width = mask.shape
    y_start = border_size
    y_end = height - border_size
    x_start = border_size
    x_end = width - border_size
    return cv2.copyMakeBorder(
        mask[y_start:y_end, x_start:x_end],
        border_size, border_size, border_size, border_size,
        cv2.BORDER_REPLICATE,
        value=None
    )


def change_gamma(mask, gamma_value):
    inv_gamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(mask, table)
