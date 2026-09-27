import unittest
from unittest.mock import patch

import torch

from matting.videomama.engine import VideoMaMaManager


class VideoMaMaDeviceTests(unittest.TestCase):
    def test_bare_cuda_device_resolves_to_current_device(self):
        with patch("torch.cuda.current_device", return_value=2):
            self.assertEqual(
                VideoMaMaManager._cuda_device_index(torch.device("cuda")), 2
            )

    def test_indexed_cuda_device_keeps_its_index(self):
        with patch("torch.cuda.current_device") as current_device:
            self.assertEqual(
                VideoMaMaManager._cuda_device_index(torch.device("cuda:3")), 3
            )
            current_device.assert_not_called()

    def test_cuda_device_index_rejects_other_device_types(self):
        with self.assertRaisesRegex(ValueError, "Expected a CUDA device"):
            VideoMaMaManager._cuda_device_index(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
