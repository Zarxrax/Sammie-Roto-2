import numpy as np
import torch

from matting.ltx_alpha.engine import LTXAlphaManager
from matting.ltx_alpha.diffusers_encoder import _CPUConv3d
from matting.ltx_alpha.gguf_runtime import (
    GGUFLinear,
    _refresh_transformer_preprocessors,
)
from matting.registry import get_engine


def test_ltx_alpha_does_not_require_segmentation():
    assert get_engine("LTXAlpha").requires_segmentation is False


def test_ltx_windows_are_adjacent_without_overlap():
    windows = LTXAlphaManager._windows(total=130, chunk=49)
    assert windows == [(0, 49), (49, 98), (98, 130)]
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))


def test_ltx_crop_expands_to_32_without_leaving_frame():
    crop = LTXAlphaManager._align_crop_to_32((101, 53, 500, 300), 1920, 1080)
    x1, y1, x2, y2 = crop
    assert (x2 - x1 + 1) % 32 == 0
    assert (y2 - y1 + 1) % 32 == 0
    assert x1 <= 101 and x2 >= 500
    assert y1 <= 53 and y2 >= 300
    assert 0 <= x1 <= x2 < 1920
    assert 0 <= y1 <= y2 < 1080


def test_ltx_crop_shifts_at_frame_edges_instead_of_clipping():
    crop = LTXAlphaManager._align_crop_to_32((0, 0, 70, 70), 1920, 1080)
    assert crop == (0, 0, 95, 95)


def test_ltx_frame_padding_uses_eight_n_plus_one():
    frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(10)]
    padded = LTXAlphaManager._pad_frames(frames)
    assert len(padded) == 17
    assert (len(padded) - 1) % 8 == 0
    assert padded[-1] is frames[-1]


def test_gguf_linear_applies_lora_without_fusing_base_weight():
    layer = GGUFLinear(3, 2, bias=False)
    layer.weight = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                                      requires_grad=False)
    layer.lora_a = torch.tensor([[0.0, 0.0, 1.0]])
    layer.lora_b = torch.tensor([[2.0], [3.0]])
    original = layer.weight.clone()

    result = layer(torch.tensor([[1.0, 2.0, 4.0]]))

    assert torch.equal(result, torch.tensor([[9.0, 14.0]]))
    assert torch.equal(layer.weight, original)


def test_refresh_transformer_preprocessors_rebinds_replaced_projection():
    class ModelType:
        @staticmethod
        def is_video_enabled(): return True

        @staticmethod
        def is_audio_enabled(): return False

    class Model:
        model_type = ModelType()

        def _init_preprocessors(self, cross_pe_max_pos):
            self.received_cross_pe_max_pos = cross_pe_max_pos
            self.preprocessor_projection = self.patchify_proj

    model = Model()
    model.patchify_proj = object()
    model.preprocessor_projection = object()

    _refresh_transformer_preprocessors(model)

    assert model.preprocessor_projection is model.patchify_proj
    assert model.received_cross_pe_max_pos is None


def test_cpu_conv3d_matches_source_convolution():
    torch.manual_seed(3)
    source = torch.nn.Conv3d(3, 4, kernel_size=3, stride=(1, 2, 2), padding=1)
    value = torch.randn(1, 3, 3, 8, 8)

    expected = source(value)
    actual = _CPUConv3d(source)(value)

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected)
