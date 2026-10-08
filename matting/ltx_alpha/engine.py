import gc
import os
import time

import numpy as np
import torch
from PySide6.QtWidgets import QApplication

from matting.base import MattingManager
from matting.ltx_alpha.downloads import ensure_models, paths
from sammie import core, image_ops
from sammie.gui_widgets import show_message_dialog
from sammie.settings_manager import get_settings_manager


class LTXAlphaManager(MattingManager):
    BACKEND = "LTXAlpha"

    def __init__(self):
        super().__init__()
        self.pipeline = None

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        if load_to_cpu:
            return False
        try:
            if not ensure_models(parent_window):
                return False
            transformer, video_vae, lora, empty_context = paths()
            from matting.ltx_alpha.pipeline import LTXAlphaPipeline
            self.pipeline = LTXAlphaPipeline(
                transformer=transformer,
                video_vae=video_vae,
                lora=lora,
                empty_context=empty_context,
                device=self._prepare_device(False),
            )
            return True
        except Exception as exc:
            print(f"Failed to load LTX 2.5 Alpha Gen: {exc}")
            if parent_window is not None:
                show_message_dialog(
                    parent_window,
                    title="LTX 2.5 Alpha Gen",
                    message=str(exc),
                    type="warning",
                )
            self.pipeline = None
            return False

    def unload_matting_model(self):
        self.pipeline = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded LTX 2.5 Alpha Gen")

    @staticmethod
    def _windows(total, chunk):
        """Split the source into adjacent chunks without shared frames."""
        return [(start, min(start + chunk, total)) for start in range(0, total, chunk)]

    @staticmethod
    def _align_crop_to_32(crop_rect, frame_w, frame_h):
        """Expand an inclusive crop so each dimension is a multiple of 32."""
        if crop_rect is None:
            return None

        def aligned_axis(low, high, limit):
            size = high - low + 1
            if size >= limit:
                return 0, limit - 1
            target = min(limit, ((size + 31) // 32) * 32)
            center = (low + high) / 2.0
            start = int(round(center - (target - 1) / 2.0))
            start = max(0, min(start, limit - target))
            return start, start + target - 1

        x1, y1, x2, y2 = crop_rect
        x1, x2 = aligned_axis(x1, x2, frame_w)
        y1, y2 = aligned_axis(y1, y2, frame_h)
        return x1, y1, x2, y2

    @staticmethod
    def _pad_frames(frames):
        required = max(1, ((len(frames) - 1 + 7) // 8) * 8 + 1)
        return frames + [frames[-1]] * (required - len(frames))

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        del combined
        if self.pipeline is None:
            return 0
        run_started = time.perf_counter()
        settings = get_settings_manager()
        start_frame, end_frame, frame_count = self._get_frame_range()
        chunk = settings.get_session_setting("ltx_alpha_chunk_frames", 49)
        crop_mode = settings.get_session_setting("ltx_alpha_crop_mode", "chunk")
        seed = settings.get_session_setting("ltx_alpha_seed", 1234)
        if (chunk - 1) % 8:
            raise ValueError("LTX Alpha chunk size must be 8n+1")

        extension = core.get_frame_extension()
        windows = self._windows(frame_count, chunk)
        progress, pbar = self._make_progress_dialog(parent_window, len(windows), unit="chunk")
        os.makedirs(core.matting_dir, exist_ok=True)
        fps = float(core.VideoInfo.fps or 24.0)
        object_ids = core.native_segmentation_object_ids(points_list or ())
        global_crop = None
        if object_ids and crop_mode == "global":
            progress.setLabelText("Finding global segmentation crop...")
            QApplication.processEvents()
            global_crop = self._align_crop_to_32(
                core.compute_mask_bounding_box(
                    range(start_frame, end_frame + 1), object_ids),
                core.VideoInfo.width,
                core.VideoInfo.height,
            )
            if global_crop is not None:
                x1, y1, x2, y2 = global_crop
                print(f"LTX Alpha global crop: ({x1}, {y1}) -> ({x2}, {y2}) "
                      f"[{x2 - x1 + 1}x{y2 - y1 + 1}]")

        try:
            for window_index, (relative_start, relative_end) in enumerate(windows):
                if progress.wasCanceled():
                    return 0
                crop_rect = global_crop
                if object_ids and crop_mode == "chunk":
                    crop_rect = self._align_crop_to_32(
                        core.compute_mask_bounding_box(
                            range(start_frame + relative_start, start_frame + relative_end),
                            object_ids,
                        ),
                        core.VideoInfo.width,
                        core.VideoInfo.height,
                    )
                    if crop_rect is not None:
                        x1, y1, x2, y2 = crop_rect
                        print(f"LTX Alpha chunk {window_index + 1}/{len(windows)} crop: "
                              f"({x1}, {y1}) -> ({x2}, {y2}) "
                              f"[{x2 - x1 + 1}x{y2 - y1 + 1}]")
                images = []
                original_size = None
                processing_size = None
                for relative_frame in range(relative_start, relative_end):
                    absolute_frame = start_frame + relative_frame
                    frame = image_ops.imread(core.frame_path(absolute_frame, extension))
                    if frame is None:
                        raise FileNotFoundError(f"Frame {absolute_frame} could not be loaded")
                    original_size = frame.shape[1], frame.shape[0]
                    if crop_rect is not None:
                        frame = core.apply_crop(frame, crop_rect)
                    processing_size = frame.shape[1], frame.shape[0]
                    rgb = image_ops.cvtColor(frame, image_ops.COLOR_BGR2RGB)
                    images.append(self._resize_image(rgb))
                padded = self._pad_frames(images)
                alpha = self.pipeline(padded, fps=fps, seed=seed + window_index)[:len(images)]

                for local_index, matte in enumerate(alpha):
                    relative_frame = relative_start + local_index
                    absolute_frame = start_frame + relative_frame
                    matte = self._restore_image_size(matte, processing_size)
                    if crop_rect is not None:
                        matte = core.expand_to_full(
                            matte, crop_rect, original_size[0], original_size[1])
                    matte = np.clip(matte * 255.0, 0, 255).astype(np.uint8)
                    output = core.output_path(core.matting_dir, absolute_frame, 0)
                    image_ops.imwrite(output, matte)
                pbar.update(1)
                progress.setValue(int((window_index + 1) / len(windows) * 100))
                QApplication.processEvents()
        finally:
            pbar.close()
            progress.close()
            elapsed = time.perf_counter() - run_started
            hours, remainder = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(remainder, 60)
            print(
                f"LTX Alpha total elapsed time: "
                f"{hours:02d}:{minutes:02d}:{seconds:02d} ({elapsed:.1f} seconds)"
            )

        if frame_count == core.VideoInfo.total_frames:
            self.propagated = True
        self._notify("matting_complete")
        return 1
