"""Image I/O through OpenImageIO and array operations used by the app.

The public functions keep the existing BGR convention at the boundary of older
call sites. OpenImageIO itself always receives and returns RGB channel order.
"""
from pathlib import Path
from functools import lru_cache

import OpenImageIO as oiio
import numpy as np

IMREAD_GRAYSCALE = 0
INTER_NEAREST = 0
INTER_LINEAR = 1
INTER_AREA = 2
COLOR_BGR2RGB = 0
COLOR_RGB2BGR = 1
COLOR_RGB2GRAY = 2
COLOR_RGBA2BGRA = 3
MORPH_ELLIPSE = 0
MORPH_GRADIENT = 1
BORDER_REPLICATE = 0
CC_STAT_AREA = 4
DIST_L2 = 0
IMWRITE_PNG_COMPRESSION = 0


@lru_cache(maxsize=1)
def supported_extensions():
    """Extensions registered by this OpenImageIO build."""
    groups = oiio.get_string_attribute("extension_list").split(";")
    return {f".{extension.lower()}" for group in groups if ":" in group
            for extension in group.partition(":")[2].split(",")}


def is_supported_image(path):
    return Path(path).suffix.lower() in supported_extensions()


def _read_rgb(path, dtype=oiio.UINT8):
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise OSError(f"OpenImageIO could not open {path}: {oiio.geterror()}")
    try:
        spec = source.spec()
        pixels = source.read_image(format=dtype)
        if pixels is None:
            raise OSError(f"OpenImageIO could not read {path}: {source.geterror()}")
        pixels = np.asarray(pixels)
        if pixels.ndim == 2:
            pixels = pixels[..., None]
        names = [name.rsplit(".", 1)[-1].lower() for name in spec.channelnames]
        if all(name in names for name in ("r", "g", "b")):
            pixels = pixels[..., [names.index("r"), names.index("g"), names.index("b")]]
        elif pixels.shape[2] >= 3:
            pixels = pixels[..., :3]
        elif pixels.shape[2] <= 2:
            pixels = np.repeat(pixels[..., :1], 3, axis=2)
        return pixels
    finally:
        source.close()


def read_rgb(path, dtype=np.uint8):
    """Read an OIIO image as RGB. Float sources map 0..1 to 0..255 for uint8."""
    format_type = oiio.UINT8 if np.dtype(dtype) == np.dtype("uint8") else oiio.FLOAT
    return _read_rgb(path, format_type)


def imread(path, flags=None):
    if not Path(path).exists():
        return None
    try:
        pixels = read_rgb(path)
    except OSError:
        return None
    if flags == IMREAD_GRAYSCALE:
        return cvtColor(pixels, COLOR_RGB2GRAY)
    return pixels[..., ::-1].copy()


def write_rgb(path, pixels):
    array = np.asarray(pixels)
    if array.ndim == 2:
        array = array[..., None]
    if array.dtype == np.float16:
        format_type = oiio.HALF
    elif array.dtype == np.float32 or array.dtype == np.float64:
        format_type = oiio.FLOAT
        array = array.astype(np.float32)
    elif array.dtype == np.uint16:
        format_type = oiio.UINT16
    else:
        format_type = oiio.UINT8
        array = array.astype(np.uint8, copy=False)
    array = np.ascontiguousarray(array)
    output = oiio.ImageOutput.create(str(path))
    if output is None:
        raise OSError(f"OpenImageIO cannot write {path}: {oiio.geterror()}")
    try:
        spec = oiio.ImageSpec(array.shape[1], array.shape[0], array.shape[2], format_type)
        if not output.open(str(path), spec):
            raise OSError(f"OpenImageIO cannot open output {path}: {output.geterror()}")
        if not output.write_image(array):
            raise OSError(f"OpenImageIO cannot write pixels to {path}: {output.geterror()}")
    finally:
        output.close()
    return True


def write_exr_layers(path, layers):
    """Write named float channels as a multilayer EXR."""
    if not layers:
        raise ValueError("EXR requires at least one channel")
    names = list(layers)
    arrays = [np.asarray(layers[name], dtype=np.float32) for name in names]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("EXR layers must have the same dimensions")
    pixels = np.ascontiguousarray(np.stack(arrays, axis=-1))
    spec = oiio.ImageSpec(shape[1], shape[0], len(names), oiio.FLOAT)
    spec.channelnames = names
    spec.attribute("compression", "zip")
    output = oiio.ImageOutput.create(str(path))
    if output is None:
        raise OSError(f"OpenImageIO cannot write {path}: {oiio.geterror()}")
    try:
        if not output.open(str(path), spec):
            raise OSError(f"OpenImageIO cannot open output {path}: {output.geterror()}")
        if not output.write_image(pixels):
            raise OSError(f"OpenImageIO cannot write pixels to {path}: {output.geterror()}")
    finally:
        output.close()


def imwrite(path, pixels, params=None):
    array = np.asarray(pixels)
    if array.ndim == 3 and array.shape[2] in (3, 4):
        array = array[..., [2, 1, 0] + ([3] if array.shape[2] == 4 else [])]
    return write_rgb(path, array)


def _image_buffer(image):
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    pixel_type = oiio.FLOAT if array.dtype.kind == "f" else oiio.UINT8
    if pixel_type == oiio.FLOAT:
        array = np.ascontiguousarray(array, dtype=np.float32)
    else:
        array = np.ascontiguousarray(array, dtype=np.uint8)
    height, width, channels = array.shape
    buffer = oiio.ImageBuf(oiio.ImageSpec(width, height, channels, pixel_type))
    roi = oiio.ROI(0, width, 0, height, 0, 1, 0, channels)
    if not buffer.set_pixels(roi, array):
        raise RuntimeError(buffer.geterror())
    return buffer, pixel_type


def _buffer_pixels(buffer, pixel_type, original):
    pixels = np.asarray(buffer.get_pixels(pixel_type))
    if original.ndim == 2:
        pixels = pixels[..., 0]
    return pixels.astype(original.dtype, copy=False)


def resize(image, size, interpolation=INTER_LINEAR):
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("resize dimensions must be positive")
    source, pixel_type = _image_buffer(image)
    channels = 1 if image.ndim == 2 else image.shape[2]
    roi = oiio.ROI(0, width, 0, height, 0, 1, 0, channels)
    if interpolation == INTER_NEAREST:
        result = oiio.ImageBufAlgo.resample(source, False, roi=roi)
    else:
        filter_name = "box" if interpolation == INTER_AREA else "triangle"
        result = oiio.ImageBufAlgo.resize(source, filtername=filter_name, roi=roi)
    if result.has_error:
        raise RuntimeError(result.geterror())
    return _buffer_pixels(result, pixel_type, image)


def cvtColor(image, code):
    if code in (COLOR_BGR2RGB, COLOR_RGB2BGR):
        return image[..., ::-1].copy()
    if code == COLOR_RGBA2BGRA:
        return image[..., [2, 1, 0, 3]].copy()
    if code == COLOR_RGB2GRAY:
        rgb = image[..., :3].astype(np.float32)
        gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
        return np.clip(np.rint(gray), 0, 255).astype(image.dtype)
    raise ValueError(f"Unknown color conversion: {code}")


def merge(channels):
    return np.stack(channels, axis=-1)


def bitwise_or(a, b):
    return np.bitwise_or(a, b)


def bitwise_and(a, b, mask=None):
    result = np.bitwise_and(a, b)
    if mask is not None:
        result = np.where((mask != 0)[..., None] if result.ndim == 3 else mask != 0, result, 0)
    return result


def countNonZero(image):
    return int(np.count_nonzero(image))


def addWeighted(a, alpha, b, beta, gamma):
    result = a.astype(np.float32) * alpha + b.astype(np.float32) * beta + gamma
    return np.clip(np.rint(result), 0, 255).astype(a.dtype)


def blendLinear(a, b, weight_a, weight_b):
    if np.ndim(weight_a) == 2 and a.ndim == 3:
        weight_a = weight_a[..., None]
    if np.ndim(weight_b) == 2 and b.ndim == 3:
        weight_b = weight_b[..., None]
    result = a.astype(np.float32) * weight_a + b.astype(np.float32) * weight_b
    return np.clip(np.rint(result), 0, 255).astype(a.dtype)


def getStructuringElement(shape, size):
    width, height = size
    yy, xx = np.ogrid[:height, :width]
    return ((xx - (width - 1) / 2) ** 2 / max((width / 2) ** 2, 1)
            + (yy - (height - 1) / 2) ** 2 / max((height / 2) ** 2, 1) <= 1)


def dilate(image, kernel, iterations=1):
    result = np.asarray(image)
    footprint = np.asarray(kernel, bool)
    for _ in range(iterations):
        if np.all(footprint):
            source, pixel_type = _image_buffer(result)
            output = oiio.ImageBufAlgo.dilate(source, footprint.shape[1], footprint.shape[0])
            result = _buffer_pixels(output, pixel_type, result)
        else:
            result = _morph_numpy(result, footprint, maximum=True)
    return result


def erode(image, kernel, iterations=1):
    result = np.asarray(image)
    footprint = np.asarray(kernel, bool)
    for _ in range(iterations):
        if np.all(footprint):
            source, pixel_type = _image_buffer(result)
            output = oiio.ImageBufAlgo.erode(source, footprint.shape[1], footprint.shape[0])
            result = _buffer_pixels(output, pixel_type, result)
        else:
            result = _morph_numpy(result, footprint, maximum=False)
    return result


def _morph_numpy(image, footprint, maximum):
    height, width = footprint.shape
    pad_y, pad_x = height // 2, width // 2
    border = 0 if maximum else np.iinfo(image.dtype).max if image.dtype.kind in "ui" else np.inf
    pad = ((pad_y, pad_y), (pad_x, pad_x)) + (((0, 0),) if image.ndim == 3 else ())
    padded = np.pad(image, pad, mode="constant", constant_values=border)
    output = np.full_like(image, border)
    for y, x in np.argwhere(footprint):
        part = padded[y:y + image.shape[0], x:x + image.shape[1]]
        if maximum:
            np.maximum(output, part, out=output)
        else:
            np.minimum(output, part, out=output)
    return output


def morphologyEx(image, operation, kernel):
    if operation == MORPH_GRADIENT:
        return dilate(image, kernel) - erode(image, kernel)
    raise ValueError(f"Unknown morphology operation: {operation}")


def GaussianBlur(image, size, sigma):
    if sigma <= 0:
        sigma = 0.3 * ((size[0] - 1) * 0.5 - 1) + 0.8
    radius = max(size) // 2
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    work = image.astype(np.float32)
    for axis in (0, 1):
        padding = [(0, 0)] * work.ndim
        padding[axis] = (radius, radius)
        extended = np.pad(work, padding, mode="reflect")
        filtered = np.zeros_like(work)
        for index, weight in enumerate(weights):
            slices = [slice(None)] * work.ndim
            slices[axis] = slice(index, index + work.shape[axis])
            filtered += weight * extended[tuple(slices)]
        work = filtered
    return np.clip(np.rint(work), 0, 255).astype(image.dtype) if image.dtype.kind in "ui" else work.astype(image.dtype)


def circle(image, center, radius, color, thickness):
    x, y = center
    y0, y1 = max(0, y - radius - 1), min(image.shape[0], y + radius + 2)
    x0, x1 = max(0, x - radius - 1), min(image.shape[1], x + radius + 2)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = (xx - x) ** 2 + (yy - y) ** 2
    region = distance <= radius ** 2
    if thickness > 0:
        region &= distance >= max(0, radius - thickness) ** 2
    image[y0:y1, x0:x1][region] = color
    return image


def connectedComponentsWithStats(mask, connectivity=8):
    """Label 2-D mask runs with NumPy and union overlapping adjacent rows."""
    foreground = np.asarray(mask) != 0
    height, width = foreground.shape
    labels = np.zeros((height, width), np.int32)
    parents = [0]
    runs = []
    previous = []
    padding = 1 if connectivity == 8 else 0

    def find(label):
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return label

    for y, row in enumerate(foreground):
        edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        current = []
        previous_index = 0
        for start, end in zip(starts, ends):
            start, end = int(start), int(end)
            label = len(parents)
            parents.append(label)
            while previous_index < len(previous) and previous[previous_index][1] <= start - padding:
                previous_index += 1
            neighbor_index = previous_index
            while neighbor_index < len(previous) and previous[neighbor_index][0] < end + padding:
                neighbor = previous[neighbor_index][2]
                left, right = find(label), find(neighbor)
                if left != right:
                    parents[right] = left
                neighbor_index += 1
            labels[y, start:end] = label
            run = (start, end, label)
            current.append(run)
            runs.append((y, start, end, label))
        previous = current

    roots = [find(label) for label in range(1, len(parents))]
    root_ids = {root: index for index, root in enumerate(sorted(set(roots)), start=1)}
    mapping = np.zeros(len(parents), np.int32)
    for label, root in enumerate(roots, start=1):
        mapping[label] = root_ids[root]
    labels = mapping[labels]
    count = len(root_ids)
    stats = np.zeros((count + 1, 5), np.int32)
    centers = np.zeros((count + 1, 2), np.float64)
    stats[1:, 0:2] = (width, height)
    sum_x = np.zeros(count + 1, np.float64)
    sum_y = np.zeros(count + 1, np.float64)
    for y, start, end, old_label in runs:
        label = mapping[old_label]
        length = end - start
        stats[label, 0] = min(stats[label, 0], start)
        stats[label, 1] = min(stats[label, 1], y)
        stats[label, 2] = max(stats[label, 2], end)
        stats[label, 3] = max(stats[label, 3], y + 1)
        stats[label, CC_STAT_AREA] += length
        sum_x[label] += (start + end - 1) * length / 2
        sum_y[label] += y * length
    stats[1:, 2] -= stats[1:, 0]
    stats[1:, 3] -= stats[1:, 1]
    stats[0, CC_STAT_AREA] = height * width - int(stats[1:, CC_STAT_AREA].sum())
    valid = stats[:, CC_STAT_AREA] > 0
    centers[valid, 0] = sum_x[valid] / stats[valid, CC_STAT_AREA]
    centers[valid, 1] = sum_y[valid] / stats[valid, CC_STAT_AREA]
    return count + 1, labels, stats, centers


def distanceTransform(mask, distance_type, mask_size):
    """Exact Euclidean distance to a zero pixel via two 1-D lower envelopes."""
    foreground = np.pad(np.asarray(mask) != 0, 1)
    height, width = foreground.shape
    far = float(height * height + width * width)
    squared = np.where(foreground, far, 0.0)
    for y in range(height):
        squared[y] = _distance_transform_1d(squared[y])
    for x in range(width):
        squared[:, x] = _distance_transform_1d(squared[:, x])
    return np.sqrt(squared[1:-1, 1:-1]).astype(np.float32)


def _distance_transform_1d(values):
    length = len(values)
    sites = np.empty(length, np.int32)
    boundaries = np.empty(length + 1, np.float64)
    sites[0] = 0
    boundaries[0] = -np.inf
    boundaries[1] = np.inf
    top = 0
    for index in range(1, length):
        while True:
            site = sites[top]
            crossing = ((values[index] + index * index)
                        - (values[site] + site * site)) / (2 * (index - site))
            if crossing > boundaries[top]:
                break
            top -= 1
        top += 1
        sites[top] = index
        boundaries[top] = crossing
        boundaries[top + 1] = np.inf
    output = np.empty(length, np.float64)
    top = 0
    for index in range(length):
        while boundaries[top + 1] < index:
            top += 1
        delta = index - sites[top]
        output[index] = delta * delta + values[sites[top]]
    return output


def copyMakeBorder(image, top, bottom, left, right, border_type, value=None):
    pad = ((top, bottom), (left, right)) + (((0, 0),) if image.ndim == 3 else ())
    return np.pad(image, pad, mode="edge")


def LUT(image, table):
    return table[image]


def inpaint(image, mask):
    """OIIO push-pull fill for masked pixels in an RGB image."""
    rgb = image[..., ::-1].astype(np.float32) / 255.0
    alpha = (mask == 0).astype(np.float32)
    rgb *= alpha[..., None]
    rgba = np.concatenate((rgb, alpha[..., None]), axis=2)
    buffer = oiio.ImageBuf(oiio.ImageSpec(image.shape[1], image.shape[0], 4, oiio.FLOAT))
    if not buffer.set_pixels(oiio.ROI(0, image.shape[1], 0, image.shape[0], 0, 1, 0, 4), rgba):
        raise RuntimeError(buffer.geterror())
    filled = oiio.ImageBufAlgo.fillholes_pushpull(buffer).get_pixels(oiio.FLOAT)
    result = np.clip(np.rint(filled[..., :3] * 255), 0, 255).astype(np.uint8)
    return result[..., ::-1].copy()
