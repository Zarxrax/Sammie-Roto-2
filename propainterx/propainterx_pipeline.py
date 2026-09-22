# sammie/propainterx_pipeline.py
"""
In-process wrapper around ProPainterX (https://github.com/Zarxrax/ProPainterX).

This is a refactor of ProPainterX's `inference_propainter.py` __main__ script into a
reusable, importable pipeline. It intentionally drops several things the original
script supports, to keep this integration simple:

  - RAFT flow estimation is not supported. MemFOF is the only flow model.
  - Video outpainting mode is not supported (object removal / inpainting only).
  - The script's own `--native_resolution_composite` compositing is not used —
    sammie already composites the (possibly downscaled) processed frame back over
    the original full-resolution frame itself (see RemovalManager.composite_removal_over_original),
    so ProPainterX is always run at whatever resolution the frames/masks are passed in at.
  - No masked-input preview video, no --vram_report diagnostics.

Everything else (chunking strategy for optical flow / flow completion / image
propagation / the transformer stage, fp16 autocast behavior, reference-frame
sampling) mirrors the original script's defaults as closely as possible.
"""
import gc
import os
import sys

import numpy as np
import torch


def _ensure_propainterx_on_syspath():
    """
    ProPainterX's own files (model/propainter.py, model/recurrent_flow_completion.py,
    model/modules/flow_comp_memfof.py, etc.) import each other with absolute imports
    like `from model.modules.deformconv import ...`, written on the assumption that the
    repo root itself sits on sys.path (true when running inference_propainter.py
    directly from that folder). This file lives inside that same propainterx/ folder,
    so we just need to make sure this folder itself is on sys.path for those internal
    `model.xxx` imports to resolve, regardless of how this module was imported.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)


_ensure_propainterx_on_syspath()

from model.modules.flow_comp_memfof import MEMFOF_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator

import cv2

# scipy.ndimage.binary_dilation's default structuring element (connectivity=1, 2D) is
# a 3x3 cross; cv2.dilate with a MORPH_CROSS kernel and the same iteration count
# produces an identical result without pulling in scipy as a dependency.
_DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


class CancelledError(Exception):
    """Raised internally when the caller's should_cancel() callback returns True."""
    pass


class _ForceOffline:
    """
    Context manager that forces HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE while MemFOF is
    constructed, restoring whatever was there before on exit. This is on top of (not a
    replacement for) requiring a local snapshot directory below — belt and suspenders
    against any library-internal 'check for updates' network call, so a run can never
    silently reach out to the Hub.
    """
    _KEYS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

    def __enter__(self):
        self._previous = {k: os.environ.get(k) for k in self._KEYS}
        for k in self._KEYS:
            os.environ[k] = "1"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self._previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ProPainterXPipeline:
    """
    Loads ProPainterX's models once and can run inpainting on multiple clips.

    Usage:
        pipe = ProPainterXPipeline(device, propainter_ckpt, flowcomp_ckpt, memfof_model_dir)
        pipe.load()
        output_frames = pipe.run(frames, masks, ...)
        pipe.unload()

    `frames` is a list of HxWx3 uint8 RGB numpy arrays.
    `masks` is a list of HxW uint8 numpy arrays (0 = keep, 255 = remove), one per frame,
    already resized to match `frames` and already aligned to a multiple of 8 in both
    dimensions (the caller is responsible for resizing/alignment).

    Returns a list of HxWx3 uint8 RGB numpy arrays, same length as the input.

    This pipeline is fully offline: it never falls back to downloading MemFOF (or
    anything else) from the Hugging Face Hub. `memfof_model_dir` must be a local
    directory already containing that model's config + weights (e.g. as produced by
    `huggingface-cli download <repo id> --local-dir <memfof_model_dir>`, or your own
    downloader); the pipeline raises rather than silently reaching out to the network.
    """

    MEMFOF_CHUNK_LEN = 8
    USE_SHARED_FNET = True
    MASK_DILATION = 4
    REF_STRIDE = 10
    NEIGHBOR_LENGTH = 10
    SUBVIDEO_LENGTH = 80
    IMG_PROPAGATION_CHUNK_SIZE = 80
    FLOW_COMPLETION_CHUNK_SIZE = 40
    ENCODER_CHUNK_SIZE = 10

    def __init__(self, device, propainter_ckpt, flowcomp_ckpt, memfof_model_dir, fp16=True, clear_cache=None):
        if not memfof_model_dir or not os.path.isdir(memfof_model_dir):
            raise FileNotFoundError(
                f"ProPainterX: memfof_model_dir '{memfof_model_dir}' was not found. This "
                f"pipeline is offline-only and will not download MemFOF from the Hugging "
                f"Face Hub — point memfof_model_dir at a local directory containing that "
                f"model's config.json and weights."
            )

        self.device = device
        self.propainter_ckpt = propainter_ckpt
        self.flowcomp_ckpt = flowcomp_ckpt
        self.memfof_model_dir = memfof_model_dir
        # RAFT's correlation volume is unstable in fp16; that restriction doesn't apply
        # to MemFOF, and CPU inference is always fp32 regardless of this flag.
        self.use_half = bool(fp16) and device.type != "cpu"
        self._clear_cache = clear_cache or self._default_clear_cache

        self.fix_flow_complete = None
        self.model = None

    def load(self):
        """Load the flow-completion and ProPainter models. MemFOF is loaded per-run
        (it's comparatively cheap to construct and this keeps run() self-contained)."""
        self.fix_flow_complete = RecurrentFlowCompleteNet(
            self.flowcomp_ckpt, encoder_chunk_size=self.ENCODER_CHUNK_SIZE
        )
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device)
        self.fix_flow_complete.eval()

        self.model = InpaintGenerator(
            model_path=self.propainter_ckpt, encoder_chunk_size=self.ENCODER_CHUNK_SIZE
        ).to(self.device)
        self.model.eval()

    def unload(self):
        self.fix_flow_complete = None
        self.model = None
        gc.collect()
        self._clear_cache()
    def _default_clear_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _dilate_mask(mask_uint8, iterations):
        """mask_uint8: HxW, {0, 255}. Returns {0, 1} uint8, dilated (mirrors
        ProPainterX's read_mask(), where flow_mask_dilates == mask_dilates by default,
        so a single dilated mask serves both roles)."""
        if iterations > 0:
            m = cv2.dilate((mask_uint8 > 0).astype(np.uint8), _DILATE_KERNEL, iterations=iterations)
        else:
            m = (mask_uint8 > 25).astype(np.uint8)
        return m

    @staticmethod
    def _to_device_tuple(t, device):
        return (t[0].to(device), t[1].to(device))

    def _get_ref_index(self, mid_neighbor_id, neighbor_ids, length, ref_stride, ref_num):
        ref_index = []
        if ref_num == -1:
            for i in range(0, length, ref_stride):
                if i not in neighbor_ids:
                    ref_index.append(i)
        else:
            start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
            end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
            for i in range(start_idx, end_idx, ref_stride):
                if i not in neighbor_ids:
                    if len(ref_index) > ref_num:
                        break
                    ref_index.append(i)
        return ref_index

    def _check_cancel(self, should_cancel):
        if should_cancel is not None and should_cancel():
            raise CancelledError("cancelled by user")

    _STAGES = [
        "Running optical flow estimation...",
        "Completing optical flow...",
        "Propagating images...",
        "Running inpainting transformer...",
    ]

    def _report(self, on_progress, should_cancel, stage_idx, done, total):
        """Report progress within stage `stage_idx` (0-based, into _STAGES) as
        `done`/`total` chunks completed in that stage alone — each stage's progress
        is independent (0..1 per stage), not blended into one overall percentage."""
        self._check_cancel(should_cancel)
        if on_progress is None:
            return
        frac = min((done / total) if total else 1.0, 1.0)
        on_progress(stage_idx, len(self._STAGES), self._STAGES[stage_idx], done, total)

    # ------------------------------------------------------------------ main entry

    def run(self, frames, masks, on_progress=None, on_frame_done=None, should_cancel=None):
        """
        frames: list of HxWx3 uint8 RGB numpy arrays (already sized/aligned to /8).
        masks: list of HxW uint8 numpy arrays, {0, 255}, same length as frames.
        on_progress: optional callable(stage_idx: int, stage_count: int, label: str,
                     done: int, total: int) called as work progresses within a stage.
                     stage_idx/stage_count let the caller show "stage 2 of 4"; done/total
                     is that stage's own chunk progress (independent per stage, not part
                     of one blended overall percentage).
        on_frame_done: optional callable(index: int, frame: np.ndarray) called during the
                       transformer stage as soon as each output frame reaches its final
                       pixel values (see note below), so the caller can save it to disk
                       immediately instead of waiting for the whole clip to finish. If
                       provided, run() returns None instead of the full frame list.
        should_cancel: optional callable() -> bool, checked periodically; raises
                       CancelledError if it returns True.

        Returns: list of HxWx3 uint8 RGB numpy arrays if on_frame_done was not given,
                 else None (frames were already delivered via on_frame_done).

        Note on on_frame_done timing: the transformer stage predicts overlapping
        sliding windows of frames and blends overlapping predictions 50/50, so a given
        frame's pixels aren't final until every window touching it has been processed.
        This is tracked internally (see `safe_upto` below) — on_frame_done only fires
        once a frame's value is truly final, in the same order the algorithm would have
        produced if the whole clip were buffered and returned at the end.
        """
        device = self.device
        use_half = self.use_half

        h, w = frames[0].shape[:2]
        video_length = len(frames)
        ori_frames = list(frames)  # shallow copy of the list; entries freed incrementally below

        # ---- build tensors ----
        # frames: (1, T, 3, H, W) uint8 [0,255]; masks_dilated: (1, T, 1, H, W) uint8 {0,1}
        frames_t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0).contiguous()
        dilated = [self._dilate_mask(m, self.MASK_DILATION) for m in masks]
        masks_dilated = torch.from_numpy(np.stack(dilated)[:, None, :, :]).unsqueeze(0).contiguous()
        # Flow masks use the same dilation amount by default, so they're identical here.
        flow_masks = masks_dilated

        ##############################################
        # ---- stage 1: optical flow (MemFOF only) ----
        ##############################################
        self._report(on_progress, should_cancel, 0, 0, 1)
        with _ForceOffline():
            fix_raft = MEMFOF_bi(self.memfof_model_dir, device, use_shared_fnet=self.USE_SHARED_FNET)
        flow_iters = 6
        short_clip_len = self.MEMFOF_CHUNK_LEN

        stage1_autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_half)
        with torch.inference_mode(), stage1_autocast_ctx:
            if video_length > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                stage1_chunks = list(range(0, video_length, short_clip_len))
                for chunk_idx, f in enumerate(stage1_chunks):
                    self._report(on_progress, should_cancel, 0, chunk_idx, len(stage1_chunks))
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        chunk = frames_t[:, f:end_f].to(device)
                    else:
                        chunk = frames_t[:, f - 1:end_f].to(device)
                    chunk = chunk.float() / 255 * 2 - 1
                    flows_f, flows_b = fix_raft(chunk, iters=flow_iters)
                    gt_flows_f_list.append(flows_f.cpu())
                    gt_flows_b_list.append(flows_b.cpu())
                    del chunk, flows_f, flows_b
                    self._clear_cache()
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
                del gt_flows_f_list, gt_flows_b_list
            else:
                chunk = frames_t.to(device).float() / 255 * 2 - 1
                flows_f, flows_b = fix_raft(chunk, iters=flow_iters)
                gt_flows_bi = (flows_f.cpu(), flows_b.cpu())
                del chunk, flows_f, flows_b
                self._clear_cache()
        del fix_raft
        self._clear_cache()
        self._report(on_progress, should_cancel, 0, 1, 1)

        ##############################################
        # ---- stage 2: flow completion ----
        ##############################################
        self._report(on_progress, should_cancel, 1, 0, 1)
        stage2_autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_half)
        with torch.inference_mode(), stage2_autocast_ctx:
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > self.FLOW_COMPLETION_CHUNK_SIZE:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                stage2_chunks = list(range(0, flow_length, self.FLOW_COMPLETION_CHUNK_SIZE))
                for chunk_idx, f in enumerate(stage2_chunks):
                    self._report(on_progress, should_cancel, 1, chunk_idx, len(stage2_chunks))
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + self.FLOW_COMPLETION_CHUNK_SIZE + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + self.FLOW_COMPLETION_CHUNK_SIZE)
                    flows_chunk = (gt_flows_bi[0][:, s_f:e_f].to(device), gt_flows_bi[1][:, s_f:e_f].to(device))
                    masks_chunk = flow_masks[:, s_f:e_f + 1].to(device).float()
                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(flows_chunk, masks_chunk)
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(flows_chunk, pred_flows_bi_sub, masks_chunk)
                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e].cpu())
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e].cpu())
                    del flows_chunk, masks_chunk, pred_flows_bi_sub
                    self._clear_cache()
                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
                del pred_flows_f, pred_flows_b
            else:
                flows_all = self._to_device_tuple(gt_flows_bi, device)
                masks_all = flow_masks.to(device).float()
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(flows_all, masks_all)
                pred_flows_bi = self.fix_flow_complete.combine_flow(flows_all, pred_flows_bi, masks_all)
                pred_flows_bi = (pred_flows_bi[0].cpu(), pred_flows_bi[1].cpu())
                del flows_all, masks_all
                self._clear_cache()
        del gt_flows_bi, flow_masks
        self._clear_cache()
        self._report(on_progress, should_cancel, 1, 1, 1)

        ##############################################
        # ---- stage 3a: image propagation ----
        ##############################################
        self._report(on_progress, should_cancel, 2, 0, 1)
        subvideo_length_img_prop = self.IMG_PROPAGATION_CHUNK_SIZE
        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_half)
        with torch.inference_mode(), autocast_ctx:
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                stage3a_chunks = list(range(0, video_length, subvideo_length_img_prop))
                for chunk_idx, f in enumerate(stage3a_chunks):
                    self._report(on_progress, should_cancel, 2, chunk_idx, len(stage3a_chunks))
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    flows_chunk = (pred_flows_bi[0][:, s_f:e_f - 1].to(device), pred_flows_bi[1][:, s_f:e_f - 1].to(device))
                    frames_chunk = frames_t[:, s_f:e_f].float() / 255 * 2 - 1
                    masks_chunk_cpu = masks_dilated[:, s_f:e_f].float()
                    masked_chunk = (frames_chunk * (1 - masks_chunk_cpu)).to(device)
                    masks_chunk = masks_chunk_cpu.to(device)
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(
                        masked_chunk, flows_chunk, masks_chunk, "nearest"
                    )
                    prop_imgs_sub = prop_imgs_sub.cpu()
                    updated_local_masks_sub = updated_local_masks_sub.cpu()
                    updated_frames_sub = frames_chunk * (1 - masks_chunk_cpu) + \
                        prop_imgs_sub.view(b, t, 3, h, w) * masks_chunk_cpu
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                    del flows_chunk, frames_chunk, masks_chunk_cpu, masked_chunk, masks_chunk, prop_imgs_sub, updated_local_masks_sub
                    self._clear_cache()
                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                flows_all = self._to_device_tuple(pred_flows_bi, device)
                frames_f = frames_t.float() / 255 * 2 - 1
                masks_f = masks_dilated.float()
                masked_all = (frames_f * (1 - masks_f)).to(device)
                masks_all = masks_f.to(device)
                prop_imgs, updated_local_masks = self.model.img_propagation(masked_all, flows_all, masks_all, "nearest")
                updated_frames = frames_f * (1 - masks_f) + prop_imgs.cpu().view(b, t, 3, h, w) * masks_f
                updated_masks = updated_local_masks.cpu().view(b, t, 1, h, w)
                del flows_all, frames_f, masks_f, masked_all, masks_all, prop_imgs, updated_local_masks
                self._clear_cache()
        del frames_t
        self._clear_cache()
        self._report(on_progress, should_cancel, 2, 1, 1)

        ##############################################
        # ---- stage 3b: transformer (feature propagation) ----
        ##############################################
        self._report(on_progress, should_cancel, 3, 0, 1)
        comp_frames = [None] * video_length
        neighbor_stride = self.NEIGHBOR_LENGTH // 2
        max_ref_frames = self.SUBVIDEO_LENGTH // self.REF_STRIDE
        ref_num = max_ref_frames if video_length > self.SUBVIDEO_LENGTH else -1

        # A frame at index `idx` is touched by every window whose center f satisfies
        # |idx - f| <= neighbor_stride. Windows are processed in increasing order of f,
        # so once we've processed the window at f, every idx <= f - neighbor_stride can
        # no longer be touched by any future window and is safe to flush/free.
        last_flushed = -1

        def _flush_up_to(safe_upto):
            nonlocal last_flushed
            idx = last_flushed + 1
            while idx <= safe_upto and idx < video_length:
                if on_frame_done is not None and comp_frames[idx] is not None:
                    on_frame_done(idx, comp_frames[idx])
                    comp_frames[idx] = None
                    ori_frames[idx] = None
                last_flushed = idx
                idx += 1

        autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_half)
        with torch.inference_mode(), autocast_ctx:
            stage3b_chunks = list(range(0, video_length, neighbor_stride))
            for chunk_idx, f in enumerate(stage3b_chunks):
                self._report(on_progress, should_cancel, 3, chunk_idx, len(stage3b_chunks))
                neighbor_ids = [
                    i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))
                ]
                ref_ids = self._get_ref_index(f, neighbor_ids, video_length, self.REF_STRIDE, ref_num)
                selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :].to(device)
                selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :].to(device).float()
                selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :].to(device)
                selected_pred_flows_bi = (
                    pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :].to(device),
                    pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :].to(device),
                )
                l_t = len(neighbor_ids)

                pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.cpu().view(-1, 3, h, w)
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated[0, neighbor_ids, :, :, :].permute(0, 2, 3, 1).numpy().astype(np.uint8)

                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + \
                        ori_frames[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)

                del selected_imgs, selected_masks, selected_update_masks, selected_pred_flows_bi, pred_img
                self._clear_cache()

                self._check_cancel(should_cancel)
                _flush_up_to(f - neighbor_stride)

        # No more windows will run; everything left is now final.
        _flush_up_to(video_length - 1)

        del pred_flows_bi, updated_frames, updated_masks, masks_dilated
        self._clear_cache()
        self._report(on_progress, should_cancel, 3, 1, 1)

        if on_frame_done is not None:
            return None
        return comp_frames
