"""Regression checks for the OpenImageIO image boundary."""
import tempfile
import unittest
from pathlib import Path

import OpenImageIO as oiio
import numpy as np
import torch

from sammie import core, image_ops
from segmentation.sam2.vendor.utils.misc import load_video_frames


class ImageOpsTests(unittest.TestCase):
    def test_exr_float_input_maps_to_model_rgb_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.exr"
            source = np.full((6, 8, 3), 0.5, np.float32)
            image_ops.write_rgb(path, source)
            self.assertIn(".exr", image_ops.supported_extensions())
            preview = image_ops.read_rgb(path)
            self.assertEqual(preview.shape, (6, 8, 3))
            self.assertEqual(int(preview[0, 0, 0]), 128)
            native = image_ops.read_rgb(path, np.float32)
            self.assertAlmostEqual(float(native[0, 0, 0]), 0.5)

    def test_multilayer_exr_keeps_channel_names(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "masks.exr"
            image_ops.write_exr_layers(path, {
                "Object_0.Y": np.full((4, 5), 0.25, np.float32),
                "Object_1.Y": np.full((4, 5), 0.75, np.float32),
            })
            source = oiio.ImageInput.open(str(path))
            try:
                self.assertEqual(tuple(source.spec().channelnames), ("Object_0.Y", "Object_1.Y"))
                pixels = source.read_image(format=oiio.FLOAT)
                self.assertAlmostEqual(float(pixels[0, 0, 1]), 0.75)
            finally:
                source.close()

    def test_push_pull_fill_uses_surrounding_pixels(self):
        frame = np.full((20, 20, 3), 50, np.uint8)
        frame[7:13, 7:13] = 200
        mask = np.zeros((20, 20), np.uint8)
        mask[7:13, 7:13] = 255
        result = image_ops.inpaint(frame, mask)
        self.assertLess(abs(int(result[10, 10, 0]) - 50), 5)
        self.assertEqual(int(result[0, 0, 0]), 50)

    def test_sam2_async_loader_reads_exr_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "00000.exr"
            image_ops.write_rgb(path, np.full((8, 10, 3), 0.5, np.float32))
            frames, height, width = load_video_frames(
                directory, 16, True, async_loading_frames=True,
                compute_device=torch.device("cpu"),
            )
            self.assertEqual((height, width), (8, 10))
            self.assertEqual(tuple(frames[0].shape), (3, 16, 16))

    def test_numpy_mask_components_and_hole_fill(self):
        mask = np.zeros((20, 20), np.uint8)
        mask[3:17, 3:17] = 255
        mask[9:11, 9:11] = 0
        filled = core.fill_small_holes(mask, 2)
        self.assertEqual(int(filled[9, 9]), 255)
        self.assertEqual(int(filled[0, 0]), 0)
        count, _, stats, _ = image_ops.connectedComponentsWithStats(mask != 0)
        self.assertEqual(count, 2)
        self.assertEqual(int(stats[1, image_ops.CC_STAT_AREA]), 192)


if __name__ == "__main__":
    unittest.main()
