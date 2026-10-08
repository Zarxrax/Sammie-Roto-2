"""GGUF Q4 loader for the vendored LTX runtime.

The packed base weight stays quantized.  A Linear dequantizes only its own
weight for the duration of a forward call and evaluates LoRA as ``B(A(x))``;
the adapter is never fused into a full-size transformer tensor.
"""

from __future__ import annotations

import copy
import logging
import warnings

import gguf
import torch
from torch import nn
from torch.nn import functional as F

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.loader.helpers import create_meta_model
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sft_loader import SafetensorsStateDictLoader
from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator
from ltx_core.text_encoders.gemma.embeddings_connector import Embeddings1DConnectorConfigurator

from .gguf_dequant import dequantize_tensor, is_quantized

logger = logging.getLogger(__name__)


class _MPSBFloatAttention:
    """Run Apple's memory efficient fused attention with BF16 inputs."""

    def __init__(self):
        from ltx_core.model.transformer.attention import MPSSdpaAttention
        self._attention = MPSSdpaAttention()

    def __call__(self, q, k, v, heads, mask=None):
        output_dtype = q.dtype
        if mask is None:
            attention_mask = None
        else:
            limit = torch.finfo(torch.bfloat16).max
            attention_mask = mask.clamp(min=-limit, max=limit).to(torch.bfloat16)
        return self._attention(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            heads, attention_mask,
        ).to(output_dtype)


# Exact metadata from the public LTX-2.5 22B distilled transformer. GGUF keeps
# tensor metadata but does not carry the LTX runtime's safetensors config blob.
_TRANSFORMER_CONFIG = {
    "_class_name": "AVTransformer3DModel",
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 128,
    "attention_type": "default",
    "caption_channels": 3840,
    "cross_attention_dim": 4096,
    "double_self_attention": False,
    "dropout": 0.0,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-6,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 48,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "cross_attention_norm": True,
    "out_channels": 128,
    "upcast_attention": False,
    "use_linear_projection": False,
    "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000,
    "av_ca_timestep_scale_multiplier": 1000.0,
    "causal_temporal_positioning": True,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "use_audio_video_cross_attention": True,
    "ff_bias": False,
    "share_ff": False,
    "audio_out_channels": 128,
    "audio_cross_attention_dim": 2048,
    "audio_positional_embedding_max_pos": [20],
    "av_cross_ada_norm": True,
    "use_embeddings_connector": True,
    "connector_attention_head_dim": 128,
    "connector_num_attention_heads": 32,
    "connector_num_layers": 8,
    "connector_positional_embedding_max_pos": [4096],
    "connector_num_learnable_registers": 128,
    "connector_norm_output": True,
    "use_middle_indices_grid": True,
    "apply_gated_attention": True,
    "connector_apply_gated_attention": True,
    "caption_projection_first_linear": False,
    "caption_projection_second_linear": False,
    "caption_proj_input_norm": False,
    "connector_learnable_registers_std": 1,
    "caption_proj_before_connector": True,
    "audio_connector_attention_head_dim": 64,
    "audio_connector_num_attention_heads": 32,
    "cross_attention_adaln": True,
    "rope_type": "split",
    "frequencies_precision": "float64",
    "text_encoder_norm_type": "PER_TOKEN_RMS",
    "use_keyframes_abs_pos_embedding": True,
}
_METADATA = {
    "model_version": "2.5",
    "gemma_source_checkpoint": {"ltx_version": "2.5.0", "gemma_version": "gemma4-12b-ltx-v1"},
    "config": {
        "transformer": _TRANSFORMER_CONFIG,
        "scheduler": {"_class_name": "RectifiedFlowScheduler", "num_train_timesteps": 1000,
                      "shifting": None, "base_resolution": None, "sampler": "LinearQuadratic"},
    },
}


class GGMLTensor(torch.Tensor):
    """Packed GGUF storage with the logical unquantized shape."""

    @staticmethod
    def __new__(cls, data, *, tensor_type, tensor_shape):
        value = torch.Tensor._make_subclass(cls, data, require_grad=False)
        value.tensor_type = tensor_type
        value.tensor_shape = torch.Size(tensor_shape)
        return value

    @property
    def shape(self):
        return getattr(self, "tensor_shape", self.size())

    def to(self, *args, **kwargs):
        qtype = getattr(self, "tensor_type", None)
        packed = qtype not in (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)
        if packed:
            args = list(args)
            if args and isinstance(args[0], torch.dtype):
                args.pop(0)
            elif len(args) > 1 and isinstance(args[1], torch.dtype):
                args.pop(1)
            kwargs.pop("dtype", None)
            args = tuple(args)
        data = super().to(*args, **kwargs)
        return GGMLTensor(data.as_subclass(torch.Tensor),
                          tensor_type=qtype,
                          tensor_shape=getattr(self, "tensor_shape", data.size()))

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self


def _orig_shape(reader, name):
    field = reader.get_field(f"comfy.gguf.orig_shape.{name}")
    if field is not None:
        return torch.Size(int(field.parts[i][0]) for i in field.data)
    tensor = next(t for t in reader.tensors if t.name == name)
    return torch.Size(int(v) for v in reversed(tensor.shape))


class GGUFStateDictLoader:
    def metadata(self, path: str) -> dict:  # noqa: ARG002
        return copy.deepcopy(_METADATA)

    def load(self, path, sd_ops=None, device=None) -> StateDict:
        paths = path if isinstance(path, list) else [path]
        if len(paths) != 1:
            raise ValueError("The LTX GGUF transformer must be a single file")
        reader = gguf.GGUFReader(paths[0], mode="r")
        prefix = "model.diffusion_model."
        has_prefix = any(t.name.startswith(prefix) for t in reader.tensors)
        result = {}
        size = 0
        dtypes = set()
        for tensor in reader.tensors:
            raw_name = tensor.name
            # Prompt connector weights are bundled in this GGUF for workflows
            # that encode text. Sammie supplies the already processed empty
            # context, so these modules are intentionally not part of LTXModel.
            if raw_name.startswith(("video_embeddings_connector.", "audio_embeddings_connector.")):
                continue
            if has_prefix and not raw_name.startswith(prefix):
                continue
            name = raw_name[len(prefix):] if has_prefix else raw_name
            # ComfyUI GGUF files without the raw prefix already use the final
            # LTX module names. Prefixed conversion outputs still need the
            # official Comfy-to-LTX mapping.
            effective_ops = sd_ops if has_prefix else None
            if effective_ops is not None:
                name = effective_ops.apply_to_key(name)
            if name is None:
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
                packed = torch.from_numpy(tensor.data)
            shape = _orig_shape(reader, raw_name)
            if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
                value = packed.view(shape)
            else:
                value = GGMLTensor(packed, tensor_type=tensor.tensor_type, tensor_shape=shape)
                if len(shape) <= 1:
                    value = dequantize_tensor(value, dtype=torch.float32)
            pairs = ((name, value),) if effective_ops is None else effective_ops.apply_to_key_value(name, value)
            for final_name, final_value in pairs:
                result[final_name] = final_value
                size += final_value.untyped_storage().nbytes()
                dtypes.add(final_value.dtype)
        logger.info("Loaded %d GGUF transformer tensors (%.2f GiB packed)",
                    len(result), size / 2**30)
        return StateDict(result, torch.device(device or "cpu"), size, dtypes)


class GGUFLinear(nn.Linear):
    """Linear with on-demand GGUF dequantization and an unfused LoRA term."""

    def __init__(self, in_features, out_features, bias=True, *, device=None, dtype=None):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None
        self.register_buffer("lora_a", None, persistent=False)
        self.register_buffer("lora_b", None, persistent=False)
        self.lora_strength = 1.0
        self.runtime_dtype = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight = state_dict.pop(prefix + "weight", None)
        bias = state_dict.pop(prefix + "bias", None)
        if weight is None:
            missing_keys.append(prefix + "weight")
        else:
            self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)

    def forward(self, value):
        output_dtype = self.runtime_dtype or value.dtype
        # PyTorch's MPS BF16 GEMM currently returns FP16 for some matrix
        # shapes. That silently changes every following activation to FP16 and
        # overflows this 22B transformer. Keep packed storage unchanged, but do
        # each dequantized linear calculation in FP32 on Metal and cast its
        # result back to the model dtype.
        compute_dtype = torch.float32 if value.device.type == "mps" else output_dtype
        compute_value = value.to(dtype=compute_dtype)
        weight = self.weight
        if is_quantized(weight):
            weight = dequantize_tensor(weight, dtype=compute_dtype, dequant_dtype=compute_dtype)
            if isinstance(weight, GGMLTensor):
                weight = weight.as_subclass(torch.Tensor)
        else:
            weight = weight.to(dtype=compute_dtype)
        bias = None if self.bias is None else self.bias.to(dtype=compute_dtype)
        output = F.linear(compute_value, weight, bias)
        if self.lora_a is not None:
            output.add_(F.linear(F.linear(compute_value, self.lora_a.to(compute_dtype)),
                                 self.lora_b.to(compute_dtype)), alpha=self.lora_strength)
        return output.to(dtype=output_dtype)


def _replace_linears(module):
    for name, child in tuple(module.named_children()):
        if isinstance(child, nn.Linear):
            replacement = GGUFLinear(child.in_features, child.out_features, child.bias is not None,
                                     device="meta", dtype=child.weight.dtype)
            setattr(module, name, replacement)
        else:
            _replace_linears(child)
    return module


def _refresh_transformer_preprocessors(model):
    """Rebind helper objects that captured Linear modules before replacement.

    LTX constructs its argument preprocessors while the model is still on the
    meta device.  They are plain Python objects, so replacing registered Linear
    children does not update their stored module references.
    """
    cross_pe_max_pos = None
    if model.model_type.is_video_enabled() and model.model_type.is_audio_enabled():
        cross_pe_max_pos = max(
            model.positional_embedding_max_pos[0],
            model.audio_positional_embedding_max_pos[0],
        )
    model._init_preprocessors(cross_pe_max_pos)


def _first_nonfinite_tensor(value):
    if isinstance(value, torch.Tensor):
        return value if value.is_floating_point() and not torch.isfinite(value).all() else None
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_nonfinite_tensor(item)
            if found is not None:
                return found
    if hasattr(value, "__dataclass_fields__"):
        for field_name in value.__dataclass_fields__:
            found = _first_nonfinite_tensor(getattr(value, field_name))
            if found is not None:
                return found
    return None


def _install_first_block_finite_trace(model):
    """Identify the first failing MPS operation, then remove all trace hooks."""
    handles = []

    def check(name):
        def hook(_module, _inputs, output):
            invalid = _first_nonfinite_tensor(output)
            if invalid is not None:
                finite = invalid[torch.isfinite(invalid)]
                finite_range = (
                    "no finite values" if finite.numel() == 0
                    else f"finite range {finite.min().item():.6g}..{finite.max().item():.6g}"
                )
                raise RuntimeError(
                    f"LTX Alpha first produced non-finite values in {name} "
                    f"({finite_range}, dtype={invalid.dtype}, shape={tuple(invalid.shape)})."
                )
        return hook

    for name, module in model.named_modules():
        if name.startswith("transformer_blocks.0.") or (
            not name.startswith("transformer_blocks.")
            and name.startswith(("patchify_proj", "adaln_single", "prompt_adaln_single"))
        ):
            handles.append(module.register_forward_hook(check(name)))
            if name == "patchify_proj":
                handles.append(module.register_forward_pre_hook(
                    lambda _module, inputs: check("patchify_proj input")(
                        _module, (), inputs)))

    def finish_first_block(_module, _inputs, _output):
        for handle in handles:
            handle.remove()

    handles.append(model.transformer_blocks[0].register_forward_hook(finish_first_block))


def _module_for_key(model, key):
    module = model
    for part in key.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _load_connector(checkpoint: str, device: torch.device, dtype: torch.dtype):
    """Build only the video prompt connector stored beside the GGUF transformer."""
    reader = gguf.GGUFReader(checkpoint, mode="r")
    prefix = "video_embeddings_connector."
    state = {}
    for tensor in reader.tensors:
        if not tensor.name.startswith(prefix):
            continue
        name = tensor.name[len(prefix):]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            packed = torch.from_numpy(tensor.data)
        shape = _orig_shape(reader, tensor.name)
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            value = packed.view(shape)
        else:
            value = GGMLTensor(packed, tensor_type=tensor.tensor_type, tensor_shape=shape)
            if len(shape) <= 1:
                value = dequantize_tensor(value, dtype=torch.float32)
        state[name] = value

    connector = _replace_linears(create_meta_model(Embeddings1DConnectorConfigurator, _METADATA))
    if device.type != "mps":
        from ltx_core.model.transformer.attention import Attention, PytorchAttention
        attention = PytorchAttention()
        for module in connector.modules():
            if isinstance(module, Attention):
                module.attention_function = attention
                module.masked_attention_function = attention
    missing, unexpected = connector.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Incomplete GGUF video connector (missing={missing[:10]}, unexpected={unexpected[:10]})")
    return connector.to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def process_cached_conditioning(checkpoint: str, conditioning: torch.Tensor,
                                device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert ComfyUI's tiny projected empty prompt into final video tokens."""
    if conditioning.ndim != 3 or conditioning.shape[-1] != 6144:
        raise ValueError(
            "Expected lossless LTX 2.5 conditioning shaped [batch, tokens, 6144], "
            f"got {tuple(conditioning.shape)}")
    video_features = conditioning[..., :4096].to(device=device, dtype=dtype)
    connector = _load_connector(checkpoint, device, dtype)
    register_count = connector.num_learnable_registers or 1
    # Match ComfyUI's LTXAVModel.preprocess_text_embeds exactly: connector
    # inputs are extended to at least 1024 tokens, using repeated learned
    # registers after the real prompt tokens.
    padded_tokens = max(
        1024,
        ((video_features.shape[1] + register_count - 1) // register_count) * register_count,
    )
    valid_tokens = video_features.shape[1]
    if padded_tokens != valid_tokens:
        video_features = F.pad(video_features, (0, 0, 0, padded_tokens - valid_tokens))
    additive_mask = torch.full(
        (video_features.shape[0], 1, 1, padded_tokens),
        -torch.finfo(dtype).max, device=device, dtype=dtype)
    additive_mask[..., :valid_tokens] = 0
    encoded, encoded_mask = connector(video_features, additive_mask)
    valid = (encoded_mask < 0.000001).to(encoded.dtype).reshape(
        encoded.shape[0], encoded.shape[1], 1)
    result = (encoded * valid).cpu()
    del connector, video_features, additive_mask, encoded, encoded_mask
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result


class GGUFTransformerBuilder:
    """LTX ModelBuilder that never materializes a full precision base state dict."""

    def __init__(self, checkpoint: str, lora: str, strength: float = 1.0):
        self._checkpoint = checkpoint
        self._lora = lora
        self._strength = strength
        self._loader = GGUFStateDictLoader()
        self._module_ops = ()

    @property
    def checkpoint(self): return self._checkpoint
    @property
    def model_path(self): return self._checkpoint
    @property
    def model_sd_ops(self): return LTXV_MODEL_COMFY_RENAMING_MAP
    @property
    def model_loader(self): return self._loader
    @property
    def module_ops(self): return self._module_ops
    @property
    def loras(self): return ()
    @property
    def registry(self): return None
    @property
    def lora_load_device(self): return torch.device("cpu")
    @property
    def fuse_rule(self): return None
    @property
    def keeps_gpu_resident_weights(self): return False

    def model_metadata(self): return self._loader.metadata(self._checkpoint)
    def model_config(self): return self.model_metadata()["config"]

    def with_module_ops(self, ops):
        result = copy.copy(self)
        result._module_ops = ops
        return result

    def with_loras(self, loras):
        if loras:
            raise ValueError("GGUF builder accepts the Alpha Gen LoRA supplied at construction")
        return self

    def with_registry(self, registry): return self  # noqa: ARG002
    def with_lora_load_device(self, device): return self  # noqa: ARG002
    def with_fuse_rule(self, rule): return self  # noqa: ARG002
    def with_sd_ops(self, ops): return self  # noqa: ARG002

    def build(self, device=None, dtype=None, **kwargs):  # noqa: ARG002
        device = torch.device(device or "cpu")
        model = create_meta_model(LTXModelConfigurator, self.model_metadata(), self._module_ops)
        model = _replace_linears(model)
        _refresh_transformer_preprocessors(model)
        base = self._loader.load(self._checkpoint, LTXV_MODEL_COMFY_RENAMING_MAP, torch.device("cpu"))
        missing, unexpected = model.load_state_dict(base.sd, strict=False, assign=True)
        # Missing keys may include only LoRA-free meta buffers. A real missing parameter
        # is caught after adapter installation below.
        if unexpected:
            logger.warning("Unused GGUF keys: %s", unexpected[:20])

        lora_sd = SafetensorsStateDictLoader().load(
            self._lora, LTXV_LORA_COMFY_RENAMING_MAP, torch.device("cpu")).sd
        suffix = ".lora_A.weight"
        installed = 0
        for key, a in lora_sd.items():
            if not key.endswith(suffix):
                continue
            module_name = key[:-len(suffix)]
            b = lora_sd.get(module_name + ".lora_B.weight")
            if b is None:
                raise KeyError(f"Missing LoRA B tensor for {module_name}")
            module = _module_for_key(model, module_name)
            if not isinstance(module, GGUFLinear):
                raise TypeError(f"LoRA target is not a linear layer: {module_name}")
            module.lora_a = a
            module.lora_b = b
            module.lora_strength = self._strength
            installed += 1
        if installed == 0:
            raise RuntimeError("No Alpha Gen LoRA tensors matched the GGUF transformer")
        logger.info("Attached Alpha Gen LoRA to %d quantized linear layers", installed)

        meta = [name for name, value in (*model.named_parameters(), *model.named_buffers())
                if value is not None and value.device.type == "meta"]
        if meta:
            raise RuntimeError(f"GGUF checkpoint left model tensors uninitialized: {meta[:20]}")
        model = model.to(device=device, dtype=dtype)
        for module in model.modules():
            if isinstance(module, GGUFLinear):
                module.runtime_dtype = dtype
        from ltx_core.model.transformer.attention import Attention, PytorchAttention
        attention = _MPSBFloatAttention() if device.type == "mps" else PytorchAttention()
        for module in model.modules():
            if isinstance(module, Attention):
                module.attention_function = attention
                module.masked_attention_function = attention
        if device.type == "mps":
            _install_first_block_finite_trace(model)
        return model
