"""Diffusers based LTX 2.5 reference encoder for MPS."""

import gc
import sys

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open


class _CPUConv3d(nn.Module):
    """Run one numerically unstable late MPS convolution on CPU in FP32."""

    def __init__(self, source: nn.Conv3d):
        super().__init__()
        self.weight = nn.Parameter(source.weight.detach().float().cpu(), requires_grad=False)
        self.bias = (
            None if source.bias is None
            else nn.Parameter(source.bias.detach().float().cpu(), requires_grad=False)
        )
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups

    def forward(self, value):
        result = F.conv3d(
            value.float().cpu(), self.weight, self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )
        return result.to(device=value.device, dtype=torch.float32)


def _first_nonfinite(value):
    if isinstance(value, torch.Tensor):
        return value if value.is_floating_point() and not torch.isfinite(value).all() else None
    if isinstance(value, (tuple, list)):
        for item in value:
            invalid = _first_nonfinite(item)
            if invalid is not None:
                return invalid
    return None


def _install_finite_trace(encoder):
    handles = []

    def check(name):
        def hook(_module, _inputs, output):
            invalid = _first_nonfinite(output)
            if invalid is None:
                return
            finite = invalid[torch.isfinite(invalid)]
            value_range = (
                "no finite values" if finite.numel() == 0
                else f"finite range {finite.min().item():.6g}..{finite.max().item():.6g}"
            )
            raise RuntimeError(
                f"LTX Alpha MPS encoder first produced non-finite values in {name} "
                f"({value_range}, dtype={invalid.dtype}, shape={tuple(invalid.shape)})."
            )
        return hook

    for name, module in encoder.named_modules():
        if not any(module.children()):
            handles.append(module.register_forward_hook(check(name)))
    return handles


@torch.inference_mode()
def encode_reference(checkpoint, video, device, dtype):
    """Load only the video encoder, encode one clip, then release its weights."""
    for module_name in ("scipy", "matplotlib"):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "__spec__", None) is None:
            del sys.modules[module_name]
    from diffusers.loaders.single_file_utils import convert_ltx2_vae_to_diffusers
    from diffusers.models.autoencoders.autoencoder_kl_ltx2 import LTX2VideoEncoder3d

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        original = {
            key: source.get_tensor(key)
            for key in source.keys()
            if key.startswith("encoder.") or key.startswith("per_channel_statistics.")
        }
    converted = convert_ltx2_vae_to_diffusers(original)
    mean = converted.pop("latents_mean")
    std = converted.pop("latents_std")
    encoder_state = {
        key.removeprefix("encoder."): value for key, value in converted.items()
    }

    with torch.device("meta"):
        encoder = LTX2VideoEncoder3d(
            in_channels=3,
            out_channels=128,
            block_out_channels=(256, 512, 1024, 1024),
            layers_per_block=(4, 6, 4, 2, 2),
            downsample_type=("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            patch_size=4,
            patch_size_t=1,
            is_causal=True,
            spatial_padding_mode="zeros",
        )
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete Diffusers LTX encoder mapping: missing={missing}, unexpected={unexpected}"
        )
    encoder = encoder.to(device=device, dtype=dtype).eval()
    if torch.device(device).type == "mps":
        unstable = encoder.down_blocks[3].downsamplers[0].conv
        unstable.conv = _CPUConv3d(unstable.conv)
    trace_handles = _install_finite_trace(encoder) if torch.device(device).type == "mps" else []
    encoded = encoder(video.to(device=device, dtype=dtype))[:, :128]
    encoded = (encoded - mean.to(device=device, dtype=dtype).view(1, -1, 1, 1, 1)) / std.to(
        device=device, dtype=dtype).view(1, -1, 1, 1, 1)
    result = encoded
    for handle in trace_handles:
        handle.remove()
    del encoder, encoder_state, converted, original
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result
