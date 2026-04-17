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
    color_space = 1


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



# .........................................................................................
# Crop / bounding-box utilities
# .........................................................................................

def compute_mask_bounding_box(frame_range, object_ids, combine_ids=None, buffer=0.10):
    """
    Scan all mask files across a frame range and return a single crop rect that
    covers every non-black pixel in every frame, plus a proportional buffer zone.
    The rect is clamped to the frame dimensions and snapped to multiples of 8.

    Args:
        frame_range: iterable of absolute frame numbers to scan
        object_ids:  list of object IDs whose masks should be considered.
                     When combine_ids is set, object_ids is ignored and
                     combine_ids is used instead (mirrors _load_batch logic).
        combine_ids: optional list of object IDs to union per frame (combined mode)
        buffer:      fractional padding added beyond the tight bounding box,
                     relative to the cropped region's own width/height (default 0.10)

    Returns:
        (x1, y1, x2, y2) integers — pixel-inclusive crop rect aligned to multiples
        of 8, or None if no non-black pixels were found in any mask.
    """
    ids_to_scan = combine_ids if combine_ids is not None else object_ids

    global_x1 = None
    global_y1 = None
    global_x2 = None
    global_y2 = None
    frame_w = VideoInfo.width
    frame_h = VideoInfo.height

    for frame_num in frame_range:
        # Build the union mask for this frame across all relevant object IDs
        union_mask = None
        for oid in ids_to_scan:
            mask_path = os.path.join(mask_dir, f"{frame_num:05d}", f"{oid}.png")
            if not os.path.exists(mask_path):
                continue
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            union_mask = m if union_mask is None else np.maximum(union_mask, m)

        if union_mask is None or not np.any(union_mask):
            continue

        union_mask = apply_mask_postprocessing(union_mask)

        # Update frame dimensions from actual mask if VideoInfo isn't populated yet
        h, w = union_mask.shape
        if frame_w == 0:
            frame_w = w
        if frame_h == 0:
            frame_h = h

        # Find non-black pixel extents for this frame
        rows = np.any(union_mask > 0, axis=1)
        cols = np.any(union_mask > 0, axis=0)
        y1 = int(np.argmax(rows))
        y2 = int(len(rows) - 1 - np.argmax(rows[::-1]))
        x1 = int(np.argmax(cols))
        x2 = int(len(cols) - 1 - np.argmax(cols[::-1]))

        global_x1 = x1 if global_x1 is None else min(global_x1, x1)
        global_y1 = y1 if global_y1 is None else min(global_y1, y1)
        global_x2 = x2 if global_x2 is None else max(global_x2, x2)
        global_y2 = y2 if global_y2 is None else max(global_y2, y2)

        # Early exit: if the bounding box already covers >= 90% of the frame
        # in both dimensions, cropping won't save meaningful work.
        if ((global_x2 - global_x1) >= frame_w * 0.9 and (global_y2 - global_y1) >= frame_h * 0.9):
            return None

    if global_x1 is None:
        return None

    # Add buffer relative to the size of the cropped region itself,
    # with a minimum of 32px per side to ensure small objects have enough context.
    crop_w = global_x2 - global_x1
    crop_h = global_y2 - global_y1
    pad_x = max(32, int(crop_w * buffer))
    pad_y = max(32, int(crop_h * buffer))

    global_x1 = max(0, global_x1 - pad_x)
    global_y1 = max(0, global_y1 - pad_y)
    global_x2 = min(frame_w - 1, global_x2 + pad_x)
    global_y2 = min(frame_h - 1, global_y2 + pad_y)

    # Snap to multiples of 8 (expand outward to avoid clipping content)
    global_x1 = (global_x1 // 8) * 8
    global_y1 = (global_y1 // 8) * 8
    global_x2 = min(frame_w - 1, ((global_x2 + 7) // 8) * 8)
    global_y2 = min(frame_h - 1, ((global_y2 + 7) // 8) * 8)

    return (global_x1, global_y1, global_x2, global_y2)


def apply_crop(image, crop_rect):
    """
    Crop an image to the given rect.

    Args:
        image:     numpy array (H, W) or (H, W, C)
        crop_rect: (x1, y1, x2, y2) as returned by compute_mask_bounding_box

    Returns:
        Cropped numpy array.
    """
    x1, y1, x2, y2 = crop_rect
    return image[y1:y2 + 1, x1:x2 + 1]


def expand_to_full(image, crop_rect, full_w, full_h):
    """
    Paste a cropped image back into a black canvas of the original frame size.

    Args:
        image:     numpy array (H, W) or (H, W, C) — the cropped region
        crop_rect: (x1, y1, x2, y2) as returned by compute_mask_bounding_box
        full_w:    original frame width
        full_h:    original frame height

    Returns:
        Full-size numpy array with the cropped content pasted at the correct position.
    """
    x1, y1, x2, y2 = crop_rect
    if image.ndim == 3:
        canvas = np.zeros((full_h, full_w, image.shape[2]), dtype=image.dtype)
    else:
        canvas = np.zeros((full_h, full_w), dtype=image.dtype)
    canvas[y1:y2 + 1, x1:x2 + 1] = image
    return canvas

