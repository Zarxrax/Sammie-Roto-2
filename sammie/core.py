# sammie/core.py
from sammie import image_ops
import os
import json
import re
import numpy as np
import torch
import warnings
from pathlib import Path
from sammie.settings_manager import get_settings_manager

# .........................................................................................
# Global variables
# .........................................................................................

temp_dir = "temp"
frames_dir = os.path.join(temp_dir, "frames")
mask_dir = os.path.join(temp_dir, "masks")
paint_dir = os.path.join(temp_dir, "paint")
paint_enabled = True
backup_dir = os.path.join(temp_dir, "masks_backup")
matting_dir = os.path.join(temp_dir, "matting")
removal_dir = os.path.join(temp_dir, "removal")
plate_info_path = os.path.join(temp_dir, "plate_info.json")
_plate_info = None


def set_plate_info(source_name, frame_numbers, padding=4):
    """Record the original plate numbering for indexed application frames."""
    global _plate_info
    _plate_info = {
        "name": re.sub(r"[^A-Za-z0-9_.-]", "_", source_name).rstrip("._-") or "plate",
        "numbers": list(frame_numbers),
        "padding": max(4, padding),
    }
    os.makedirs(temp_dir, exist_ok=True)
    with open(plate_info_path, "w", encoding="utf-8") as stream:
        json.dump(_plate_info, stream)


def plate_info():
    global _plate_info
    if _plate_info is None:
        with open(plate_info_path, encoding="utf-8") as stream:
            _plate_info = json.load(stream)
    return _plate_info


def reload_plate_info():
    global _plate_info
    _plate_info = None
    return plate_info()


def plate_number(frame_index):
    numbers = plate_info()["numbers"]
    return numbers[frame_index] if 0 <= frame_index < len(numbers) else frame_index


def frame_path(frame_index, extension=None):
    extension = extension or get_frame_extension()
    info = plate_info()
    return os.path.join(frames_dir, f"{info['name']}_frames.{plate_number(frame_index):0{info['padding']}d}.{extension}")


def output_path(folder, frame_index, object_id=None):
    info = plate_info()
    folder_name = os.path.basename(folder)
    kind = "mask" if folder_name in ("masks", "masks_backup") else folder_name
    object_suffix = f"_obj{object_id}" if object_id not in (None, 0) else ""
    name = f"{info['name']}_{kind}{object_suffix}.{plate_number(frame_index):0{info['padding']}d}.png"
    return os.path.join(folder, name)


def output_ids(folder, frame_index):
    """List object IDs with output files for one frame."""
    ids = []
    if not os.path.isdir(folder):
        return ids
    base = os.path.basename(output_path(folder, frame_index, 0))
    stem, extension = os.path.splitext(base)
    prefix, number = stem.rsplit('.', 1)
    for name in os.listdir(folder):
        if name == base:
            ids.append(0)
        elif name.startswith(prefix + "_obj") and name.endswith("." + number + extension):
            token = name[len(prefix) + 4:-(len(number) + len(extension) + 1)]
            if token.isdigit():
                ids.append(int(token))
    return ids


def remove_output_objects(folder, frame_index, keep_ids=()):
    for object_id in output_ids(folder, frame_index):
        if object_id not in keep_ids:
            os.remove(output_path(folder, frame_index, object_id))


def native_segmentation_object_ids(points=()):
    """Objects represented by SAM points or raw SAM mask files."""
    ids = {point['object_id'] for point in points if 'object_id' in point}
    for frame_index in range(VideoInfo.total_frames):
        ids.update(output_ids(mask_dir, frame_index))
    return sorted(ids)


def segmentation_object_ids(points=()):
    """Use paint-only objects only when no native segmentation exists."""
    native_ids = native_segmentation_object_ids(points)
    if native_ids or not paint_enabled:
        return native_ids
    paint_ids = set()
    for frame_index in range(VideoInfo.total_frames):
        paint_ids.update(output_ids(paint_dir, frame_index))
    return sorted(paint_ids)


def segmentation_keyframes(object_id, points=(), start_frame=0, end_frame=None):
    """Use painted seed frames only when the project has no native segmentation."""
    end_frame = VideoInfo.total_frames - 1 if end_frame is None else end_frame
    frames = {
        point['frame'] for point in points
        if point.get('object_id') == object_id and start_frame <= point['frame'] <= end_frame
    }
    if frames:
        return sorted(frames)

    native_ids = native_segmentation_object_ids(points)
    if native_ids:
        return [
            frame for frame in range(start_frame, end_frame + 1)
            if os.path.exists(output_path(mask_dir, frame, object_id))
        ]

    if paint_enabled:
        return [
            frame for frame in range(start_frame, end_frame + 1)
            if os.path.exists(paint_path(frame, object_id))
        ]
    return []

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
    _dtype = torch.float32

    @classmethod
    def setup_device(cls):
        """Detect and set up the best available device"""
        if cls._device is not None:
            return cls._device  # already set

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("PyTorch version:", torch.__version__)

        is_rocm = "rocm" in torch.__version__.lower()

        # Prevent MIOpen from recompiling kernels on every run 
        if is_rocm:
            os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")

        # On Windows, redirect MIOpen's (AMD ROCm) db/cache files 
        # into this app's own folder instead of the user's global profile 
        if os.name == "nt" and is_rocm:
            miopen_cache_root = Path(__file__).resolve().parents[1] / ".runtime_cache" / "miopen"
            miopen_db_path = miopen_cache_root / "db"
            miopen_kernel_cache_path = miopen_cache_root / "cache"
            miopen_db_path.mkdir(parents=True, exist_ok=True)
            miopen_kernel_cache_path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MIOPEN_USER_DB_PATH", str(miopen_db_path))
            os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(miopen_kernel_cache_path))

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

        if cls._device.type != "cuda":
            print(f"Using device: {cls._device}")

        if cls._device.type == "cuda":
            is_rocm = torch.version.hip is not None
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                props = torch.cuda.get_device_properties(0)
                gpu_name = props.name

                if is_rocm:
                    print(f"ROCm Device: {gpu_name}")
                    print(f"ROCm/HIP version: {torch.version.hip}")
                    if torch.cuda.is_bf16_supported():
                        cls._dtype = torch.bfloat16
                    else:
                        cls._dtype = torch.float16
                else:
                    capability = (props.major, props.minor)
                    print(f"CUDA Device: {gpu_name}")
                    print(f"CUDA Capability: {capability[0]}.{capability[1]}")

                    # Enable bfloat16 for Ampere and newer
                    if torch.cuda.is_bf16_supported():
                        cls._dtype = torch.bfloat16
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    # Turing and newer have FP16 support
                    elif capability >= (7, 5):
                        # Don't let gtx 16 series use FP16
                        if "gtx 16" in gpu_name.lower():
                            cls._dtype = torch.float32
                        else:
                            cls._dtype = torch.float16
                    # Older NVIDIA GPUs: stay with FP32
                    else:
                        cls._dtype = torch.float32
                torch.autocast("cuda", dtype=cls._dtype).__enter__()

        elif cls._device.type == "mps":
            cls._dtype = torch.float16
            torch.autocast("mps", dtype=cls._dtype).__enter__()
        
        elif cls._device.type == "xpu":
            # A user confirmed that bf16 and fp16 gave broken masks on Arc A750
            cls._dtype = torch.float32
            torch.autocast("xpu", dtype=cls._dtype).__enter__()

        return cls._device

    @classmethod
    def get_device(cls):
        """Return the already initialized device (or setup if needed)"""
        if cls._device is None:
            return cls.setup_device()
        return cls._device

    @classmethod
    def get_dtype(cls):
        """Return the storage/compute dtype matching the active autocast setup"""
        if cls._device is None:
            cls.setup_device()
        return cls._dtype

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
            mask_filename = output_path(mask_dir, point["frame"], point["object_id"])
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
        before_count = len(self.points)
        points_to_remove = [p for p in self.points if p['frame'] == frame]
        self.points = [p for p in self.points if p['frame'] != frame]
        removed_count = before_count - len(self.points)

        if removed_count > 0:
            # Remove mask files for this frame
            for object_id in output_ids(mask_dir, frame):
                os.remove(output_path(mask_dir, frame, object_id))
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
                mask_filename = output_path(mask_dir, point["frame"], object_id)
                matting_filename = output_path(matting_dir, point["frame"], object_id)
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
    frame_filename = frame_path(frame_number, extension)
    if os.path.exists(frame_filename):
        image = image_ops.imread(frame_filename)
        return image_ops.cvtColor(image, image_ops.COLOR_BGR2RGB)
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

    # Discover objects from annotations and from the actual output files. The
    # latter is required for paint-only matting, where no point records exist.
    object_ids = list(
        {p['object_id'] for p in points if 'object_id' in p}
        | set(output_ids(folder, frame_number))
    )
    if folder == mask_dir and paint_enabled:
        object_ids = list(set(object_ids) | set(output_ids(paint_dir, frame_number)))

    # Filter by specific object ID if requested
    if object_id_filter is not None:
        object_ids = [obj_id for obj_id in object_ids if obj_id == object_id_filter]

    if not object_ids:
        return None if return_combined else {}

    individual_masks = {}

    # Load each mask file
    for object_id in object_ids:
        if folder == mask_dir:
            mask = load_segmentation_mask(frame_number, object_id)
        else:
            mask_filename = output_path(folder, frame_number, object_id)
            mask = image_ops.imread(mask_filename, image_ops.IMREAD_GRAYSCALE) if os.path.exists(mask_filename) else None
        individual_masks[object_id] = mask if mask is not None else np.zeros((VideoInfo.height, VideoInfo.width), dtype=np.uint8)

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


def paint_path(frame_number, object_id):
    return output_path(paint_dir, frame_number, object_id)


def load_segmentation_mask(frame_number, object_id):
    """Return one object's SAM mask with its optional paint edits applied."""
    path = output_path(mask_dir, frame_number, object_id)
    mask = image_ops.imread(path, image_ops.IMREAD_GRAYSCALE) if os.path.exists(path) else None
    if mask is None and paint_enabled and os.path.exists(paint_path(frame_number, object_id)):
        mask = np.zeros((VideoInfo.height, VideoInfo.width), dtype=np.uint8)
    if mask is not None and paint_enabled:
        mask = apply_paint(mask, frame_number, object_id)
    return mask


def _paint_masks(paint):
    """Decode white additions and red subtractions."""
    return (np.all(paint == (255, 255, 255), axis=2),
            np.all(paint == (255, 0, 0), axis=2))


def apply_paint(mask, frame_number, object_id):
    """Black leaves SAM intact, white adds, and red removes."""
    path = paint_path(frame_number, object_id)
    if not os.path.exists(path):
        return mask
    paint = image_ops.read_rgb(path)
    if paint.shape[:2] != mask.shape:
        return mask
    add, remove = _paint_masks(paint)
    result = mask.copy()
    result[add] = 255
    result[remove] = 0
    return result


class PaintStroke:
    """Keep one brush drag in memory and persist it only when finished."""

    def __init__(self, frame_number, object_id):
        self.path = paint_path(frame_number, object_id)
        height, width = VideoInfo.height, VideoInfo.width
        paint = image_ops.read_rgb(self.path) if os.path.exists(self.path) else None
        self.paint = paint if paint is not None and paint.shape[:2] == (height, width) else np.zeros((height, width, 3), dtype=np.uint8)

    def add_segment(self, start, end, radius, add):
        paint_segment(self.paint, start, end, radius, add)

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        image_ops.write_rgb(self.path, self.paint)


def paint_segment(paint, start, end, radius, add):
    """Rasterize a solid interpolated segment into an in-memory overlay."""
    height, width = paint.shape[:2]
    distance = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
    for step in range(distance + 1):
        x = round(start[0] + (end[0] - start[0]) * step / distance)
        y = round(start[1] + (end[1] - start[1]) * step / distance)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        if x0 < x1 and y0 < y1:
            yy, xx = np.ogrid[y0:y1, x0:x1]
            paint[y0:y1, x0:x1][(xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2] = (255, 255, 255) if add else (255, 0, 0)


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
    _, labels, stats, _ = image_ops.connectedComponentsWithStats(mask == 0, connectivity=8)
    border_labels = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    fill_labels = np.flatnonzero(stats[:, image_ops.CC_STAT_AREA] <= max_hole_area)
    fill_labels = fill_labels[~np.isin(fill_labels, border_labels)]
    small = np.isin(labels, fill_labels)
    filled_mask = mask.copy()
    filled_mask[small] = 255
    return filled_mask


def remove_small_dots(mask, dots_value):
    max_dot_area = dots_value ** 2
    num_labels, labels, stats, _ = image_ops.connectedComponentsWithStats(mask, connectivity=8)

    cleaned_mask = np.zeros_like(mask)
    for label in range(1, num_labels):  # skip background
        if stats[label, image_ops.CC_STAT_AREA] > max_dot_area:
            cleaned_mask[labels == label] = 255

    return cleaned_mask


def grow_shrink(mask, grow_value):
    kernel = np.ones((abs(grow_value) + 1, abs(grow_value) + 1), np.uint8)
    if grow_value > 0:
        return image_ops.dilate(mask, kernel, iterations=1)
    elif grow_value < 0:
        return image_ops.erode(mask, kernel, iterations=1)
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
    return image_ops.copyMakeBorder(
        mask[y_start:y_end, x_start:x_end],
        border_size, border_size, border_size, border_size,
        image_ops.BORDER_REPLICATE,
        value=None
    )


def change_gamma(mask, gamma_value):
    inv_gamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return image_ops.LUT(mask, table)



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
            m = load_segmentation_mask(frame_num, oid)
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
