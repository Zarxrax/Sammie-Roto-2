# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
build_sam.py, rewritten to not depend on hydra-core / omegaconf.

The original file used hydra.compose() + hydra.utils.instantiate() to turn a
yaml config into a model. The model classes themselves (SAM2Base, Hiera,
EfficientTAMBase, ViT, ...) have no hydra dependency -- hydra was only used to
wire them together. This file does that wiring directly in Python instead,
using the exact same parameters as the yaml configs it replaces:

    configs/sam2.1/sam2.1_hiera_b+.yaml
    configs/sam2.1/sam2.1_hiera_l.yaml
    configs/efficienttam/efficienttam_s_512x512.yaml

`build_sam2` / `build_sam2_video_predictor` keep their original signatures --
you still pass the same `config_file` string you used to pass to hydra (e.g.
"configs/sam2.1/sam2.1_hiera_b+.yaml"); it's matched by basename against the
registry below, so existing call sites don't need to change.

`hydra_overrides_extra` (free-form hydra override strings) is no longer
supported, since there's no hydra to apply them -- pass architecture overrides
as **kwargs instead (they're merged into the registry's base kwargs).

To add another model variant, add an entry to `_MODEL_REGISTRY` following the
existing entries as a template.
"""

import logging
import os

import torch

import sam2

# Check if the user is running Python from the parent directory of the sam2 repo
# (i.e. the directory where this repo is cloned into) -- this is not supported since
# it could shadow the sam2 package and cause issues.
if os.path.isdir(os.path.join(sam2.__path__[0], "sam2")):
    raise RuntimeError(
        "You're likely running Python from the parent directory of the sam2 repository "
        "(i.e. the directory where https://github.com/facebookresearch/sam2 is cloned into). "
        "This is not supported since the `sam2` Python package could be shadowed by the "
        "repository name (the repository is also named `sam2` and contains the Python package "
        "in `sam2/sam2`). Please run Python from another directory (e.g. from the repo dir "
        "rather than its parent dir, or from your home directory) after installing SAM 2."
    )

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder, ViTDetNeck
from sam2.modeling.backbones.vitdet import ViT
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.memory_encoder import CXBlock, Fuser, MaskDownSampler, MemoryEncoder
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_video_predictor import SAM2VideoPredictor, SAM2VideoPredictorVOS


HF_MODEL_ID_TO_FILENAMES = {
    "facebook/sam2-hiera-tiny": (
        "configs/sam2/sam2_hiera_t.yaml",
        "sam2_hiera_tiny.pt",
    ),
    "facebook/sam2-hiera-small": (
        "configs/sam2/sam2_hiera_s.yaml",
        "sam2_hiera_small.pt",
    ),
    "facebook/sam2-hiera-base-plus": (
        "configs/sam2/sam2_hiera_b+.yaml",
        "sam2_hiera_base_plus.pt",
    ),
    "facebook/sam2-hiera-large": (
        "configs/sam2/sam2_hiera_l.yaml",
        "sam2_hiera_large.pt",
    ),
    "facebook/sam2.1-hiera-tiny": (
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_tiny.pt",
    ),
    "facebook/sam2.1-hiera-small": (
        "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_small.pt",
    ),
    "facebook/sam2.1-hiera-base-plus": (
        "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_base_plus.pt",
    ),
    "facebook/sam2.1-hiera-large": (
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_large.pt",
    ),
}


# ---------------------------------------------------------------------------
# Shared sub-module builders (identical across the sam2.1 b+/l configs, and
# also identical to the memory_encoder used by efficienttam_s_512x512).
# ---------------------------------------------------------------------------

def _sam21_image_encoder(embed_dim, num_heads, backbone_channel_list, **hiera_overrides):
    trunk = Hiera(embed_dim=embed_dim, num_heads=num_heads, **hiera_overrides)
    neck = FpnNeck(
        position_encoding=PositionEmbeddingSine(
            num_pos_feats=256, normalize=True, scale=None, temperature=10000,
        ),
        d_model=256,
        backbone_channel_list=backbone_channel_list,
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )
    return ImageEncoder(scalp=1, trunk=trunk, neck=neck)


def _sam21_memory_attention(feat_sizes=(64, 64)):
    self_attention = RoPEAttention(
        rope_theta=10000.0, feat_sizes=list(feat_sizes), embedding_dim=256,
        num_heads=1, downsample_rate=1, dropout=0.1,
    )
    cross_attention = RoPEAttention(
        rope_theta=10000.0, feat_sizes=list(feat_sizes), rope_k_repeat=True,
        embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1, kv_in_dim=64,
    )
    layer = MemoryAttentionLayer(
        activation="relu", dim_feedforward=2048, dropout=0.1, pos_enc_at_attn=False,
        self_attention=self_attention, d_model=256, pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False, cross_attention=cross_attention,
    )
    return MemoryAttention(d_model=256, pos_enc_at_input=True, layer=layer, num_layers=4)


def _shared_memory_encoder():
    # Identical across sam2.1 (b+, l) and efficienttam_s_512x512 configs.
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64, normalize=True, scale=None, temperature=10000,
    )
    mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)
    cx_block = CXBlock(
        dim=256, kernel_size=7, padding=3, layer_scale_init_value=1e-6, use_dwconv=True,
    )
    fuser = Fuser(layer=cx_block, num_layers=2)
    return MemoryEncoder(
        out_dim=64, position_encoding=position_encoding,
        mask_downsampler=mask_downsampler, fuser=fuser,
    )


def _postprocessing_kwargs(base_kwarg_name="sam_mask_decoder_extra_args"):
    # Mirrors the "++model.sam_mask_decoder_extra_args...." hydra overrides
    # applied by the original build_sam2()/build_sam2_video_predictor() when
    # apply_postprocessing=True.
    return {
        base_kwarg_name: dict(
            dynamic_multimask_via_stability=True,
            dynamic_multimask_stability_delta=0.05,
            dynamic_multimask_stability_thresh=0.98,
        )
    }


def _video_postprocessing_kwargs():
    return dict(binarize_mask_from_pts_for_mem_enc=True, fill_hole_area=8)


# ---------------------------------------------------------------------------
# Per-config registry.
#
# Each entry provides:
#   base_cls          -- the class instantiated by build_sam2() / build_efficienttam()
#   video_cls          -- the class used by build_sam2_video_predictor(vos_optimized=False)
#   video_vos_cls       -- the class used by build_sam2_video_predictor(vos_optimized=True)
#   build_submodules() -- returns dict(image_encoder=..., memory_attention=..., memory_encoder=...)
#   base_kwargs         -- dict of all other (non-submodule) constructor kwargs,
#                           exactly matching the config's top-level `model:` keys
#                           (minus image_encoder/memory_attention/memory_encoder/_target_)
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "sam2.1_hiera_b+.yaml": dict(
        base_cls=SAM2Base,
        video_cls=SAM2VideoPredictor,
        video_vos_cls=SAM2VideoPredictorVOS,
        build_submodules=lambda: dict(
            image_encoder=_sam21_image_encoder(
                embed_dim=112, num_heads=2, backbone_channel_list=[896, 448, 224, 112],
            ),
            memory_attention=_sam21_memory_attention(),
            memory_encoder=_shared_memory_encoder(),
        ),
        base_kwargs=dict(
            num_maskmem=7,
            image_size=1024,
            sigmoid_scale_for_mem_enc=20.0,
            sigmoid_bias_for_mem_enc=-10.0,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            no_obj_embed_spatial=True,
            use_high_res_features_in_sam=True,
            multimask_output_in_sam=True,
            iou_prediction_use_sigmoid=True,
            use_obj_ptrs_in_encoder=True,
            add_tpos_enc_to_obj_ptrs=True,
            proj_tpos_enc_in_obj_ptrs=True,
            use_signed_tpos_enc_to_obj_ptrs=True,
            only_obj_ptrs_in_the_past_for_eval=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            multimask_output_for_tracking=True,
            use_multimask_token_for_obj_ptr=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_mlp_for_obj_ptr_proj=True,
            compile_image_encoder=False,
        ),
    ),
    "sam2.1_hiera_l.yaml": dict(
        base_cls=SAM2Base,
        video_cls=SAM2VideoPredictor,
        video_vos_cls=SAM2VideoPredictorVOS,
        build_submodules=lambda: dict(
            image_encoder=_sam21_image_encoder(
                embed_dim=144,
                num_heads=2,
                backbone_channel_list=[1152, 576, 288, 144],
                stages=[2, 6, 36, 4],
                global_att_blocks=[23, 33, 43],
                window_pos_embed_bkg_spatial_size=[7, 7],
                window_spec=[8, 4, 16, 8],
            ),
            memory_attention=_sam21_memory_attention(),
            memory_encoder=_shared_memory_encoder(),
        ),
        base_kwargs=dict(
            num_maskmem=7,
            image_size=1024,
            sigmoid_scale_for_mem_enc=20.0,
            sigmoid_bias_for_mem_enc=-10.0,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            no_obj_embed_spatial=True,
            use_high_res_features_in_sam=True,
            multimask_output_in_sam=True,
            iou_prediction_use_sigmoid=True,
            use_obj_ptrs_in_encoder=True,
            add_tpos_enc_to_obj_ptrs=True,
            proj_tpos_enc_in_obj_ptrs=True,
            use_signed_tpos_enc_to_obj_ptrs=True,
            only_obj_ptrs_in_the_past_for_eval=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            multimask_output_for_tracking=True,
            use_multimask_token_for_obj_ptr=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_mlp_for_obj_ptr_proj=True,
            compile_image_encoder=False,
        ),
    ),
    # From efficienttam_s_512x512.yaml. The yaml's own `_target_` names
    # sam2.modeling.efficienttam_base.EfficientTAMBase, but that module was
    # never actually added when EfficientTAM was integrated into this repo
    # (see facebookresearch/sam2 PR #640) -- it doesn't matter for
    # build_sam2_video_predictor, which always overrides model._target_ to
    # SAM2VideoPredictor/SAM2VideoPredictorVOS before instantiation, so the
    # yaml's own _target_ is never resolved. SAM2Base is used here as the
    # base_cls (for build_sam2()'s non-video path) since it's unmodified and
    # already generic over the image_encoder/memory_attention/memory_encoder
    # you hand it -- there's no EfficientTAM-specific subclass in this repo.
    #
    # neck_norm="LN" matches the yaml's `neck_norm: LN` -- ViTDetNeck only
    # checks `is not None`, so any truthy value enables the LayerNorm2d
    # layers; the string itself isn't parsed.
    "efficienttam_s_512x512.yaml": dict(
        base_cls=SAM2Base,
        video_cls=SAM2VideoPredictor,
        video_vos_cls=SAM2VideoPredictorVOS,
        build_submodules=lambda: dict(
            image_encoder=ImageEncoder(
                scalp=0,
                trunk=ViT(
                    patch_size=16,
                    embed_dim=384,
                    depth=12,
                    num_heads=6,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    drop_path_rate=0.0,
                    use_rel_pos=False,
                    window_size=14,
                    window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
                ),
                neck=ViTDetNeck(
                    position_encoding=PositionEmbeddingSine(
                        num_pos_feats=256, normalize=True, scale=None, temperature=10000,
                    ),
                    d_model=256,
                    backbone_channel_list=[384],
                    neck_norm="LN",
                ),
            ),
            memory_attention=_sam21_memory_attention(feat_sizes=(32, 32)),
            memory_encoder=_shared_memory_encoder(),
        ),
        base_kwargs=dict(
            num_maskmem=7,
            image_size=512,
            sigmoid_scale_for_mem_enc=20.0,
            sigmoid_bias_for_mem_enc=-10.0,
            use_mask_input_as_output_without_sam=True,
            directly_add_no_mem_embed=True,
            use_high_res_features_in_sam=False,
            multimask_output_in_sam=True,
            iou_prediction_use_sigmoid=True,
            use_obj_ptrs_in_encoder=True,
            add_tpos_enc_to_obj_ptrs=False,
            only_obj_ptrs_in_the_past_for_eval=True,
            pred_obj_scores=True,
            pred_obj_scores_mlp=True,
            fixed_no_obj_ptr=True,
            multimask_output_for_tracking=True,
            use_multimask_token_for_obj_ptr=True,
            multimask_min_pt_num=0,
            multimask_max_pt_num=1,
            use_mlp_for_obj_ptr_proj=True,
            compile_image_encoder=False,  # matches the uploaded yaml (official upstream yaml has this True)
        ),
    ),
}


def _lookup(config_file):
    key = os.path.basename(config_file)
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"No baked-in config for {config_file!r} (looked up by basename {key!r}). "
            f"Known configs: {sorted(_MODEL_REGISTRY.keys())}. "
            "Add an entry to _MODEL_REGISTRY in build_sam.py to support it."
        )
    return _MODEL_REGISTRY[key]


def _build_model(
    config_file,
    target_cls,
    ckpt_path,
    device,
    mode,
    hydra_overrides_extra,
    apply_postprocessing,
    postproc_kwarg_updates,
    extra_kwargs,
):
    if hydra_overrides_extra:
        raise ValueError(
            "hydra_overrides_extra is not supported without hydra. Pass architecture "
            "overrides as **kwargs instead (they're merged over the registry's base "
            f"kwargs); got: {hydra_overrides_extra!r}"
        )

    spec = _lookup(config_file)
    submodules = spec["build_submodules"]()
    kwargs = dict(spec["base_kwargs"])
    if apply_postprocessing:
        kwargs.update(_postprocessing_kwargs())
        kwargs.update(postproc_kwarg_updates)
    kwargs.update(extra_kwargs)  # user-supplied **kwargs win over everything

    model = target_cls(**submodules, **kwargs)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def build_sam2(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    spec = _lookup(config_file)
    return _build_model(
        config_file=config_file,
        target_cls=spec["base_cls"],
        ckpt_path=ckpt_path,
        device=device,
        mode=mode,
        hydra_overrides_extra=hydra_overrides_extra,
        apply_postprocessing=apply_postprocessing,
        postproc_kwarg_updates={},
        extra_kwargs=kwargs,
    )


def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    vos_optimized=False,
    **kwargs,
):
    spec = _lookup(config_file)
    target_cls = spec["video_vos_cls"] if vos_optimized else spec["video_cls"]
    extra_kwargs = dict(kwargs)
    if vos_optimized:
        # Mirrors "++model.compile_image_encoder=True" (let the base class handle it).
        extra_kwargs.setdefault("compile_image_encoder", True)
    return _build_model(
        config_file=config_file,
        target_cls=target_cls,
        ckpt_path=ckpt_path,
        device=device,
        mode=mode,
        hydra_overrides_extra=hydra_overrides_extra,
        apply_postprocessing=apply_postprocessing,
        postproc_kwarg_updates=_video_postprocessing_kwargs() if apply_postprocessing else {},
        extra_kwargs=extra_kwargs,
    )



def _hf_download(model_id):
    from huggingface_hub import hf_hub_download

    config_name, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[model_id]
    ckpt_path = hf_hub_download(repo_id=model_id, filename=checkpoint_name)
    return config_name, ckpt_path


def build_sam2_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2(config_file=config_name, ckpt_path=ckpt_path, **kwargs)


def build_sam2_video_predictor_hf(model_id, **kwargs):
    config_name, ckpt_path = _hf_download(model_id)
    return build_sam2_video_predictor(
        config_file=config_name, ckpt_path=ckpt_path, **kwargs
    )


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
