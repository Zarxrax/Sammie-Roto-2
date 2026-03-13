"""
CorridorKey Inference Engine — adapted for Sammie-Roto 2.

Based on EZ-CorridorKey by the Corridor Crew community.
Simplified: removed logging framework, uses print statements,
removed pynvml dependency (uses torch.cuda directly).
All inference logic preserved intact.
"""

import math
import os
import time
import types

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.model_transformer import GreenFormer
from .core import color_utils as cu


def _patch_hiera_global_attention(hiera_model: nn.Module) -> int:
    """Monkey-patch MaskUnitAttention.forward on global-attention blocks.

    Hiera's MaskUnitAttention creates Q/K/V with shape
    [B, heads, num_windows, N, head_dim]. When num_windows == 1
    (global attention), this 5-D non-contiguous tensor causes PyTorch's
    SDPA to silently fall back to the VRAM-hungry math backend.

    This patch forces Q/K/V to standard 4-D contiguous tensors, enabling
    FlashAttention and dropping VRAM usage per block dramatically.
    """
    patched = 0

    for blk in hiera_model.blocks:
        attn = blk.attn

        if attn.use_mask_unit_attn:
            continue

        def _make_patched_forward(original_attn):
            def _patched_forward(self, x: torch.Tensor) -> torch.Tensor:
                B, N, _ = x.shape
                qkv = self.qkv(x)
                qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)

                if self.q_stride > 1:
                    q = q.view(
                        B, self.heads, self.q_stride, -1, self.head_dim
                    ).amax(dim=2)

                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()

                x = F.scaled_dot_product_attention(q, k, v)
                x = x.transpose(1, 2).reshape(B, -1, self.dim_out)
                x = self.proj(x)
                return x

            return types.MethodType(_patched_forward, original_attn)

        attn.forward = _make_patched_forward(attn)
        patched += 1

    return patched


class CorridorKeyEngine:
    """
    AI-powered green screen keying engine.

    Processes single frames: takes RGB + coarse mask, returns refined alpha
    + despilled foreground + composited preview.

    Architecture: Hiera Base Plus encoder → dual decoder (alpha + FG) → CNN refiner.
    """

    _VRAM_TILE_THRESHOLD_GB = 12
    VALID_OPT_MODES = ('auto', 'speed', 'lowvram')

    def __init__(self, checkpoint_path, device='cuda', img_size=2048,
                 use_refiner=True, optimization_mode='auto',
                 tile_overlap=128, on_status=None, force_tiling=None):

        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.use_refiner = use_refiner
        self.tile_overlap = tile_overlap
        self._on_status = on_status
        self._eager_model = None
        self._compiled_model = None
        self._compile_error = None

        # torch.compile deadlocks with Qt/PySide6 on Linux — always disable
        import sys
        self._use_compile = (sys.platform != 'linux'
                             and 'PySide6' not in sys.modules)

        # Tiling: explicit user choice takes priority, then auto-detect
        if force_tiling is not None:
            self.tile_size = 512 if force_tiling else 0
            mode_label = f"tiled 512x512" if force_tiling else "full frame"
        else:
            vram_gb = self._get_vram_gb()
            if 0 < vram_gb < self._VRAM_TILE_THRESHOLD_GB:
                self.tile_size = 512
                mode_label = f"tiled 512x512 (auto, {vram_gb:.1f}GB)"
            else:
                self.tile_size = 0
                mode_label = f"full frame ({vram_gb:.1f}GB)"

        print(f"CorridorKey: {mode_label}, compile={'on' if self._use_compile else 'off'}")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self.model = self._load_model()

    @staticmethod
    def _get_vram_gb() -> float:
        """Return total GPU VRAM in GB using torch.cuda."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass
        return 0.0

    def _status(self, msg: str) -> None:
        """Emit status to callback and console."""
        print(f"[CorridorKey] {msg}")
        if self._on_status:
            try:
                self._on_status(msg)
            except Exception:
                pass

    @staticmethod
    def _iter_exception_chain(exc):
        seen = set()
        current = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            yield current
            current = current.__cause__ or current.__context__

    def _has_compiled_artifacts(self) -> bool:
        refiner = getattr(self._eager_model, 'refiner', None)
        tile_kernel = getattr(refiner, '_compiled_process_tile', None) if refiner else None
        return self._compiled_model is not None or tile_kernel is not None

    def _is_compile_failure(self, exc):
        markers = (
            'triton', 'torchinductor', 'torch._inductor', 'torch._dynamo',
            'backendcompilerfailed', 'loweringexception', 'asynccompile',
            'compileworker', 'inductor',
        )
        for link in self._iter_exception_chain(exc):
            text = (type(link).__name__ + ' ' + str(link)).lower()
            if any(m in text for m in markers):
                return True
            tb = getattr(link, '__traceback__', None)
            while tb is not None:
                fn = tb.tb_frame.f_code.co_filename.lower()
                if any(m in fn for m in markers):
                    return True
                tb = tb.tb_next
        return False

    def _disable_compile(self, exc):
        self._compile_error = exc
        self._compiled_model = None
        self.model = self._eager_model
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"[CorridorKey] Compile failed, falling back to eager mode: {exc}")

    def _forward_model(self, inp_t, refiner_scale_t):
        try:
            return self.model(inp_t, refiner_scale=refiner_scale_t)
        except Exception as e:
            if not self._has_compiled_artifacts() or not self._is_compile_failure(e):
                raise
            self._disable_compile(e)
            return self.model(inp_t, refiner_scale=refiner_scale_t)

    def _load_model(self):
        self._status("Loading model architecture...")

        model = GreenFormer(
            encoder_name='hiera_base_plus_224.mae_in1k_ft_in1k',
            in_channels=4,
            img_size=self.img_size,
            use_refiner=self.use_refiner,
        )

        # Load checkpoint
        self._status("Loading checkpoint...")
        t0 = time.monotonic()
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        print(f"[CorridorKey] Checkpoint loaded: {time.monotonic() - t0:.1f}s")

        # Remap keys: strip torch.compile '_orig_mod.' prefix, resize pos_embed
        model_state = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                k = k[10:]

            if 'pos_embed' in k and k in model_state:
                if v.shape != model_state[k].shape:
                    N_src = v.shape[1]
                    C = v.shape[2]
                    grid_src = int(math.sqrt(N_src))
                    grid_dst = int(math.sqrt(model_state[k].shape[1]))
                    v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                    v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst),
                                              mode='bicubic', align_corners=False)
                    v = v_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        self._status("Loading state dict...")
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if missing:
            print(f"[CorridorKey] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[CorridorKey] Unexpected keys: {len(unexpected)}")

        # Enable TF32 for Ampere+
        torch.set_float32_matmul_precision('high')

        # Disable cuDNN benchmark to save workspace memory
        torch.backends.cudnn.benchmark = False

        # Patch Hiera attention for FlashAttention
        self._status("Patching attention blocks...")
        try:
            hiera = model.encoder.model
            n_patched = _patch_hiera_global_attention(hiera)
            print(f"[CorridorKey] Hiera attention patch: {n_patched} blocks")
        except Exception as e:
            print(f"[CorridorKey] Hiera attention patch failed: {e}")

        # Configure tiled refiner
        if self.tile_size > 0 and hasattr(model, 'refiner') and model.refiner is not None:
            model.refiner._tile_size = self.tile_size
            model.refiner._tile_overlap = self.tile_overlap
            print(f"[CorridorKey] Tiled refiner: {self.tile_size}x{self.tile_size}")

        model.to(self.device).eval()

        eager_model = model
        self._eager_model = eager_model
        self._compiled_model = None

        # torch.compile (optional, with full fallback)
        if self._use_compile:
            self._status("Compiling model (first run may take a minute)...")

            try:
                import triton  # noqa: F401
            except Exception:
                self._use_compile = False
                print("[CorridorKey] Triton unavailable, skipping torch.compile")

            if self._use_compile and self.tile_size > 0 and \
               hasattr(eager_model, 'refiner') and eager_model.refiner is not None:
                try:
                    eager_model.refiner.compile_tile_kernel()
                    print("[CorridorKey] Refiner tile kernel compiled")
                except Exception as e:
                    print(f"[CorridorKey] Tile compile failed (using eager): {e}")

            if self._use_compile:
                try:
                    self._compiled_model = torch.compile(eager_model, fullgraph=False)
                    model = self._compiled_model
                    print("[CorridorKey] torch.compile complete")
                except Exception as e:
                    self._compiled_model = None
                    print(f"[CorridorKey] torch.compile failed (using eager): {e}")

        self._status("Model ready")
        return model

    @torch.no_grad()
    def process_frame(self, image, mask_linear, refiner_scale=1.0,
                      input_is_linear=False, fg_is_straight=True,
                      despill_strength=1.0, auto_despeckle=True,
                      despeckle_size=400, despeckle_dilation=25,
                      despeckle_blur=5):
        """
        Process a single frame.

        Args:
            image: Numpy array [H, W, 3] (0.0-1.0 or 0-255). sRGB by default.
            mask_linear: Numpy array [H, W] or [H, W, 1] (0.0-1.0). Linear.
            refiner_scale: Multiplier for Refiner Deltas (default 1.0).
            input_is_linear: If True, input is Linear (EXR). If False, sRGB.
            fg_is_straight: If True, FG output is straight (unpremultiplied).
            despill_strength: 0.0 to 1.0 multiplier for despill.
            auto_despeckle: Clean up small disconnected alpha islands.
            despeckle_size: Min pixel count to keep an island.
            despeckle_dilation: Dilation for despeckle morphology.
            despeckle_blur: Blur kernel for despeckle smoothing.

        Returns:
            dict: {
                'alpha': np [H,W,1] float (0-1), despeckled alpha matte,
                'fg': np [H,W,3] float (0-1), sRGB despilled foreground,
                'comp': np [H,W,3] float (0-1), sRGB composite on checkerboard,
            }
        """
        t0 = time.monotonic()

        # 1. Input normalization
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # 2. Resize to model size
        if input_is_linear:
            img_resized_lin = cv2.resize(image, (self.img_size, self.img_size),
                                         interpolation=cv2.INTER_LINEAR)
            img_resized = cu.linear_to_srgb(img_resized_lin)
        else:
            img_resized = cv2.resize(image, (self.img_size, self.img_size),
                                     interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size),
                                   interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # 3. Normalize (ImageNet)
        img_norm = (img_resized - self.mean) / self.std

        # 4. Prepare tensor
        inp_np = np.concatenate([img_norm, mask_resized], axis=-1)
        inp_t = torch.from_numpy(
            inp_np.transpose((2, 0, 1))
        ).float().unsqueeze(0).to(self.device)

        # 5. Inference
        refiner_scale_t = inp_t.new_tensor(refiner_scale)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            out = self._forward_model(inp_t, refiner_scale_t)

        pred_alpha = out['alpha']
        pred_fg = out['fg']

        # 6. Post-process (resize back to original)
        res_alpha = pred_alpha[0].permute(1, 2, 0).float().cpu().numpy()
        res_fg = pred_fg[0].permute(1, 2, 0).float().cpu().numpy()
        res_alpha = cv2.resize(res_alpha, (w, h), interpolation=cv2.INTER_LANCZOS4)
        res_fg = cv2.resize(res_fg, (w, h), interpolation=cv2.INTER_LANCZOS4)

        if res_alpha.ndim == 2:
            res_alpha = res_alpha[:, :, np.newaxis]

        # A. Clean matte (despeckle)
        if auto_despeckle:
            processed_alpha = cu.clean_matte(
                res_alpha, area_threshold=despeckle_size,
                dilation=despeckle_dilation, blur_size=despeckle_blur
            )
        else:
            processed_alpha = res_alpha

        # B. Despill FG
        fg_despilled = cu.despill(res_fg, green_limit_mode='average',
                                  strength=despill_strength)

        # C. Composite on checkerboard (sRGB space for display)
        fg_despilled_lin = cu.srgb_to_linear(fg_despilled)
        bg_srgb = cu.create_checkerboard(w, h, checker_size=128,
                                          color1=0.15, color2=0.55)
        bg_lin = cu.srgb_to_linear(bg_srgb)

        if fg_is_straight:
            comp_lin = cu.composite_straight(fg_despilled_lin, bg_lin, processed_alpha)
        else:
            comp_lin = cu.composite_premul(fg_despilled_lin, bg_lin, processed_alpha)

        comp_srgb = cu.linear_to_srgb(comp_lin)

        print(f"[CorridorKey] Frame {h}x{w} processed in {time.monotonic() - t0:.3f}s")

        return {
            'alpha': processed_alpha,
            'fg': fg_despilled,
            'comp': comp_srgb,
        }

    def unload(self):
        """Release model and free VRAM."""
        self._compiled_model = None
        self._eager_model = None
        self.model = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("[CorridorKey] Engine unloaded")
