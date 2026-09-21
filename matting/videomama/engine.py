import cv2
import os
import numpy as np
import torch
import gc
from tqdm import tqdm
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt
from sammie import core
from sammie.settings_manager import get_settings_manager
from sammie.model_downloader import ensure_models

from matting.base import MattingManager
from matting.videomama.downloads import DOWNLOADS

class VideoMaMaManager(MattingManager):
    """
    Matting manager that uses the VideoMaMa model.

    """

    BACKEND = "VideoMaMa"

    @staticmethod
    def _limit_mps_memory(memory_gib, enabled=True):
        """Keep the baseline MPS safety ceiling, applying the user budget if enabled."""
        recommended = torch.mps.recommended_max_memory()
        limit = min(80 * 1024 ** 3, int(recommended * 0.8))
        if enabled:
            limit = min(limit, int(memory_gib * 0.95 * 1024 ** 3))
        torch.mps.set_per_process_memory_fraction(limit / recommended)
        print(f"VideoMaMa MPS allocation limit: {limit / 1024 ** 3:.1f} GiB")

    @staticmethod
    def _limit_cuda_memory(device, memory_gib, enabled=True):
        """Limit PyTorch's CUDA caching allocator on the selected device."""
        total = torch.cuda.get_device_properties(device).total_memory
        fraction = min(0.95, memory_gib * 0.95 * 1024 ** 3 / total) if enabled else 1.0
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        if enabled:
            print(f"VideoMaMa CUDA allocation limit: {total * fraction / 1024 ** 3:.1f} GiB")

    def unload_matting_model(self):
        """Unload the VideoMaMa pipeline and free VRAM"""
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, 'unet'):
                    self.pipeline.unet = None
                if hasattr(self.pipeline, 'vae'):
                    self.pipeline.vae = None
            except Exception as e:
                print(f"Warning during pipeline teardown: {e}")
            self.pipeline = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded VideoMaMa model")
        
    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        """
        Load the VideoMaMa pipeline.
        """
        from matting.videomama.pipeline_svd_mask_numpy import VideoInferencePipeline

        device = self._prepare_device(load_to_cpu)
        settings_mgr = get_settings_manager()
        matting_model = settings_mgr.get_session_setting("matany_model", "VideoMaMa")
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        overlap = settings_mgr.get_session_setting("matany_overlap", 2)
        batch_size = max(16, settings_mgr.get_session_setting("matany_chunk", 16))
        memory_gib = settings_mgr.get_session_setting("videomama_memory_gib", 32)
        memory_enabled = settings_mgr.get_session_setting("videomama_memory_enabled", True)
        combined = settings_mgr.get_session_setting("matany_combined", False)

        if not ensure_models(list(DOWNLOADS.values()), parent=parent_window):
            return False  # user cancelled or download failed

        try:
            if device.type == "mps":
                self._limit_mps_memory(memory_gib, memory_enabled)
            elif device.type == "cuda":
                self._limit_cuda_memory(device, memory_gib, memory_enabled)
            # BF16 has FP32's exponent range with half the storage. Keep the VAE
            # in FP32 while testing a smaller UNet after all-FP16 gave black frames.
            mps_bf16 = (device.type == "mps" and
                        torch.backends.mps.is_macos_or_newer(14, 0))
            unet_dtype = torch.bfloat16 if mps_bf16 else (
                torch.float32 if device.type in ("mps", "cpu") else torch.float16)
            self.pipeline = VideoInferencePipeline(
                base_model_path=os.path.join("checkpoints", "videomama"),
                unet_checkpoint_path=os.path.join("checkpoints", "videomama"),
                weight_dtype=unet_dtype,
                device=str(device),
                enable_model_cpu_offload=False,    # Not much benefit here, since the vae is a small model
                vae_encode_chunk_size=1,          # Process VAE in small chunks, increasing doesnt help anything
                attention_mode="auto",            # Use xformers if available, else SDPA
                enable_vae_tiling=False,        # Tiling VAE is not worth it
                enable_vae_slicing=True,          # Process VAE one image at a time
            )
            print(f"Loaded {matting_model} model to {device} with max size {max_size}, overlap={overlap}, batch={batch_size}, and combined={combined}")
            return True

        except Exception as e:
            print(f"Error loading VideoMaMa pipeline: {e}")
            self.pipeline = None
            return False

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        """
        Run matting on all frames, using tracked segmentation data.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog
            combined (bool): If True, union all object masks into a single combined mask
                and run matting in a single pass instead of once per object.

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """

        if self.pipeline is None:
            print("VideoMaMa model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        frame_count = core.VideoInfo.total_frames
        extension = core.get_frame_extension()
        settings_mgr = get_settings_manager()
        
        # --- VideoMaMa batch settings ---
        batch_size = max(16, settings_mgr.get_session_setting("matany_chunk", 16))   # frames per chunk sent to the model
        overlap = settings_mgr.get_session_setting("matany_overlap", 2)     # frames re-processed at each boundary for continuity
        memory_gib = settings_mgr.get_session_setting("videomama_memory_gib", 32)
        memory_enabled = settings_mgr.get_session_setting("videomama_memory_enabled", True)
        if overlap >= batch_size:
            overlap = batch_size - 1
            print(f"VideoMaMa: reducing overlap to {overlap} for {batch_size}-frame batches")
        if self.pipeline.device.type == "mps":
            self._limit_mps_memory(memory_gib, memory_enabled)
        elif self.pipeline.device.type == "cuda":
            self._limit_cuda_memory(self.pipeline.device, memory_gib, memory_enabled)
        requested_res = settings_mgr.get_session_setting("matany_res", 0)
        print(f"VideoMaMa {self.pipeline.device.type}: {self.pipeline.weight_dtype} UNet, "
              f"{self.pipeline.vae_dtype} VAE, {batch_size} frames per batch, "
              f"working short side {requested_res or 'original'}px, "
              f"memory budget {'enabled at ' + str(memory_gib) + ' GiB' if memory_enabled else 'disabled'}.")
        if overlap == 0: 
            enable_boundary_blend = False
        else: 
            enable_boundary_blend = True # blend overlap frames linearly at chunk boundaries

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get unique object IDs from points list
        object_ids = sorted(list(set(point['object_id'] for point in points_list if 'object_id' in point)))
        if not object_ids:
            print("No objects found for matting")
            return 0

        # When combined mode is requested, run a single pass using the union of all
        # object masks loaded in memory — no files are written to disk.
        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            object_ids = [0]
        else:
            combine_ids = None

        # Create matting directory if it doesn't exist
        os.makedirs(core.matting_dir, exist_ok=True)

        # If combined mode is selected, delete any existing matting files except object 0.
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        # Calculate total operations for progress tracking
        total_operations = len(self._generate_windows(frames_to_process, batch_size, overlap)) * len(object_ids)

        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations, unit="batch")
        operations_completed = 0

        try:
            # Process each object separately
            for object_id in object_ids:
                if progress_dialog.wasCanceled():
                    break

                pbar.set_description(f"Object {object_id}")
                print(f"Processing object {object_id}...")

                batches_completed = self._process_object(
                    object_id, start_frame, end_frame, overlap, batch_size, extension,
                    progress_dialog, pbar, operations_completed,
                    total_operations, parent_window, combine_ids=combine_ids,
                    enable_boundary_blend=enable_boundary_blend
                )

                if batches_completed is None:  # cancelled or failed
                    break

                operations_completed += batches_completed

        except Exception as e:
            pbar.close()
            progress_dialog.close()
            raise

        # Final cleanup
        pbar.close()
        core.DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(100)
            if frame_count == frames_to_process:
                self.propagated = True  # only set propagated to True if the entire video was processed
            else:
                self.propagated = False
            print("Matting completed")
            self._notify('matting_complete')
            return 1

    def _load_batch(self, abs_start, abs_end, object_id, extension,
                    combine_ids=None, crop_rect=None):
        """
        Load a single batch window of frames and masks from disk, resized using
        the same proportional downscaling as MatAnyone (matany_res setting).
 
        If combine_ids is provided, masks for all IDs in that list are unioned
        in memory per frame rather than loading a single object mask from disk.
 
        If crop_rect is provided, both frames and masks are cropped to that region
        before being passed through _resize_image, so the model only processes the
        relevant portion of the frame.

        Args:
            abs_start:   Absolute start frame (inclusive)
            abs_end:     Absolute end frame (exclusive)
            object_id:   Object ID for mask lookup (ignored when combine_ids is set)
            extension:   Frame file extension
            combine_ids: Optional list of object IDs to union into a single mask
            crop_rect:   Optional (x1, y1, x2, y2) from core.compute_mask_bounding_box.
                         When set, frames and masks are cropped before resizing.
 
        Returns:
            tuple: (cond_frames, mask_frames, valid)
                - cond_frames: list of np.ndarray RGB uint8 (H, W, 3)
                - mask_frames: list of np.ndarray grayscale uint8 (H, W), binarized
                - valid: False if any source frame is missing
        """
        cond_frames = []
        mask_frames = []
 
        for frame_num in range(abs_start, abs_end):
            frame_path = os.path.join(core.frames_dir, f"{frame_num:05d}.{extension}")
            if not os.path.exists(frame_path):
                print(f"Warning: Frame not found: {frame_path}")
                return [], [], False
 
            # Load and convert frame to RGB, then proportionally downscale via matany_res
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop_rect is not None:
                frame = core.apply_crop(frame, crop_rect)
            frame = self._resize_image(frame)
            resized_h, resized_w = frame.shape[:2]
 
            # Load mask — union across all combine_ids in memory, or single object from disk
            if combine_ids:
                union_mask = None
                for oid in combine_ids:
                    mask_path = os.path.join(core.mask_dir, f"{frame_num:05d}", f"{oid}.png")
                    if not os.path.exists(mask_path):
                        continue
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    union_mask = m if union_mask is None else np.maximum(union_mask, m)
                if union_mask is not None:
                    union_mask = core.apply_mask_postprocessing(union_mask)
                    if crop_rect is not None:
                        union_mask = core.apply_crop(union_mask, crop_rect)
                    mask = cv2.resize(union_mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
            else:
                mask_path = os.path.join(core.mask_dir, f"{frame_num:05d}", f"{object_id}.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = core.apply_mask_postprocessing(mask)
                    if crop_rect is not None:
                        mask = core.apply_crop(mask, crop_rect)
                    mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
 
            # Binarize mask
            mask = (mask > 127).astype(np.uint8) * 255
 
            cond_frames.append(frame)
            mask_frames.append(mask)
 
        return cond_frames, mask_frames, True

    def _generate_windows(self, total_frames, batch_size, overlap):
        """
        Generate sliding window (start, end) pairs covering total_frames.
        end is exclusive. Overlap must be less than batch_size.

        Args:
            total_frames: Total frames to cover
            batch_size: Frames per batch 
            overlap: Overlap frames between consecutive batches

        Returns:
            list of (start, end) tuples (relative indices, end exclusive)
        """
        step = batch_size - overlap
        if step <= 0:
            print(f"Warning: overlap ({overlap}) >= batch_size ({batch_size}), clamping to batch_size - 1")
            overlap = batch_size - 1
            step = 1

        if total_frames <= batch_size:
            return [(0, total_frames)]

        windows = []
        pos = 0
        while pos < total_frames:
            end = min(pos + batch_size, total_frames)
            windows.append((pos, end))
            if end >= total_frames:
                break
            pos += step

        return windows


    def _process_object(self, object_id, start_frame, end_frame, overlap, batch_size, extension,
                        progress_dialog, pbar, operations_completed, total_operations, parent_window,
                        combine_ids=None, enable_boundary_blend=True):
        """
        Process a single object across all frames using windowed batching.
 
        Args:
            object_id: Object ID to process (also the output file label)
            start_frame: First frame (inclusive)
            end_frame: Last frame (inclusive)
            overlap: Overlap frames between batches
            batch_size: Number of frames per chunk sent to the model
            extension: Frame file extension
            progress_dialog: Qt progress dialog
            pbar: tqdm progress bar
            operations_completed: Batches completed before this object (for progress math)
            total_operations: Total batch count across all objects
            parent_window: Parent window for display updates
            combine_ids: If set, union masks for these IDs in memory rather than
                         loading a single object mask from disk
            enable_boundary_blend: Linearly blend overlap frames at chunk boundaries
 
        Returns:
            int: Number of batches completed, or None if cancelled/failed
        """
        frames_to_process = end_frame - start_frame + 1
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        memory_gib = settings_mgr.get_session_setting("videomama_memory_gib", 32)
        memory_enabled = settings_mgr.get_session_setting("videomama_memory_enabled", True)
 
        # Original frame dimensions for restoring output
        first_frame_path = os.path.join(core.frames_dir, f"{start_frame:05d}.{extension}")
        first_frame_img = cv2.imread(first_frame_path)
        if first_frame_img is not None:
            original_h, original_w = first_frame_img.shape[:2]
        else:
            original_w = core.VideoInfo.width
            original_h = core.VideoInfo.height

        # Compute a single crop rect covering all mask extents across the full frame range.
        # This is done once before any batch processing so every batch uses an identical
        # spatial region, keeping output consistent across chunk boundaries.
        object_ids_for_bbox = combine_ids if combine_ids is not None else [object_id]
        progress_dialog.setLabelText(f"Object {object_id}: Finding region of interest...")
        QApplication.processEvents()

        crop_rect = core.compute_mask_bounding_box(
            frame_range=range(start_frame, end_frame + 1),
            object_ids=object_ids_for_bbox,
            combine_ids=combine_ids,
        )

        if crop_rect is not None:
            cx1, cy1, cx2, cy2 = crop_rect
            print(f"Object {object_id}: crop rect ({cx1}, {cy1}) -> ({cx2}, {cy2})  "
                  f"[{cx2 - cx1 + 1}x{cy2 - cy1 + 1} of {original_w}x{original_h}]")
        else:
            print(f"Object {object_id}: Processing full frame")
 
        windows = self._generate_windows(frames_to_process, batch_size, overlap)
        batches_completed = 0

        # Stores soft alpha mattes (model working resolution, grayscale uint8) for the
        # last `overlap` frames of each batch.  Fed back as mask_frames for the overlap
        # frames of the next batch, giving the model continuity across chunk boundaries.
        # Kept soft (not binarized) so the model sees graduated edge information.
        previous_overlap_masks = None

        # Stores the last `overlap` committed alpha mattes at original resolution,
        # used for linear blending across the boundary after each non-first batch.
        prev_boundary_alphas = None
 
        for batch_idx, (window_start, window_end) in enumerate(windows):
            if progress_dialog.wasCanceled():
                return None
 
            abs_start = start_frame + window_start
            abs_end = start_frame + window_end  # exclusive
 
            # For non-first batches, the leading overlap frames are warm-up context only —
            # their output is discarded in favour of the already-committed result from the
            # previous batch. Only the non-overlap tail is saved.
            is_first_batch = (batch_idx == 0)
            output_start_offset = 0 if is_first_batch else overlap
 
            cond_frames, mask_frames, valid = self._load_batch(
                abs_start, abs_end, object_id, extension,
                combine_ids=combine_ids, crop_rect=crop_rect
            )
 
            if not valid:
                print(f"Warning: Skipping batch {batch_idx} for object {object_id} "
                    f"(frames {abs_start}-{abs_end - 1}) — missing data")
                pbar.update(1)
                batches_completed += 1
                # Reset both carry-overs so stale data is never injected after a gap
                previous_overlap_masks = None
                prev_boundary_alphas = None
                continue
 
            # --- FEEDBACK LOOP INJECTION ---
            # Replace the SAM2 masks for the overlap frames with the soft alpha mattes
            # predicted by the previous batch.  The model sees its own prior output as
            # guidance, encouraging consistent alpha values across the chunk boundary.
            if not is_first_batch and previous_overlap_masks is not None:
                for i in range(min(overlap, len(mask_frames), len(previous_overlap_masks))):
                    mask_frames[i] = previous_overlap_masks[i]
            # -------------------------------

            def _on_pipeline_progress(step, total, desc):
                progress_dialog.setLabelText(f"Batch {batch_idx + 1}/{len(windows)} — {desc}")
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise RuntimeError("USER_CANCELLED")

            device_type = core.DeviceManager.get_device().type 
            try:
                with torch.amp.autocast(device_type=device_type, enabled=False):
                    if memory_enabled and self.pipeline.device.type in ("mps", "cuda"):
                        from matting.videomama.spatial_tiling import run_tiled
                        tile_budget = memory_gib
                        if self.pipeline.device.type == "cuda":
                            available_gib = (torch.cuda.get_device_properties(
                                self.pipeline.device).total_memory / 1024 ** 3)
                            tile_budget = max(8, min(memory_gib, available_gib * 0.95))
                        output_frames = run_tiled(
                            self.pipeline, cond_frames, mask_frames,
                            memory_gib=tile_budget, seed=42,
                            progress_callback=_on_pipeline_progress,
                        )
                    else:
                        output_frames = self.pipeline.run(
                            cond_frames=cond_frames,
                            mask_frames=mask_frames,
                            seed=42,
                            progress_callback=_on_pipeline_progress,
                        )
            except RuntimeError as e:
                if str(e) == "USER_CANCELLED":
                    return None  # signals cancel upstream
                if "out of memory" in str(e).lower():
                    print("VideoMaMa reached the GPU memory limit. Lower frames per batch "
                          "or adjust the memory budget before retrying.")
                raise
            except Exception as e:
                print(f"Error in VideoMaMa inference for batch {batch_idx}: {e}")
                raise
 
            # --- CAPTURE SOFT MASKS FOR NEXT BATCH ---
            # Store the last `overlap` frames of the current output at model resolution
            # (before _restore_image_size) as soft grayscale — NOT binarized, so the
            # model receives graduated edge values rather than a hard binary boundary.
            previous_overlap_masks = []
            for frame_out in output_frames[-overlap:]:
                alpha_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2GRAY)
                previous_overlap_masks.append(alpha_out)
            # -----------------------------------------
 
            # Discard leading overlap output on non-first batches
            committed_output = output_frames[output_start_offset:]

            # --- LINEAR BLEND AT CHUNK BOUNDARIES ---
            # The overlap frames were re-processed by the new batch, giving us a second
            # prediction for those already-committed frames (output_frames[0:overlap]).
            # We linearly blend the original committed alpha (prev_boundary_alphas) with
            # the new prediction across the overlap window and re-write those frames.
            if enable_boundary_blend and not is_first_batch and prev_boundary_alphas is not None:
                for i in range(min(overlap, len(prev_boundary_alphas), len(output_frames))):
                    new_alpha = cv2.cvtColor(output_frames[i], cv2.COLOR_RGB2GRAY)
                    new_alpha = self._restore_image_size(new_alpha, (cx2 - cx1 + 1, cy2 - cy1 + 1) if crop_rect else (original_w, original_h))
                    if crop_rect is not None:
                        new_alpha = core.expand_to_full(new_alpha, crop_rect, original_w, original_h)
                    new_weight = (i + 1) / (overlap + 1)
                    blended = (
                        (1.0 - new_weight) * prev_boundary_alphas[i].astype(np.float32)
                        + new_weight * new_alpha.astype(np.float32)
                    ).clip(0, 255).astype(np.uint8)
                    abs_blend_frame = abs_start + i
                    mat_filename = os.path.join(core.matting_dir, f"{abs_blend_frame:05d}", f"{object_id}.png")
                    os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                    cv2.imwrite(mat_filename, blended)
            # -----------------------------------------

            # Collect the last `overlap` committed alphas (original resolution) for the
            # next batch's boundary blend, then write all committed frames to disk.
            current_boundary_alphas = []
            for i, frame_out in enumerate(committed_output):
                abs_frame = abs_start + output_start_offset + i
                alpha = cv2.cvtColor(frame_out, cv2.COLOR_RGB2GRAY)

                # Restore to the cropped region's pixel dimensions first, then expand
                # back into the full frame canvas so the saved matte is always full-size.
                if crop_rect is not None:
                    crop_w = cx2 - cx1 + 1
                    crop_h = cy2 - cy1 + 1
                    alpha = self._restore_image_size(alpha, (crop_w, crop_h))
                    final_alpha = core.expand_to_full(alpha, crop_rect, original_w, original_h)
                else:
                    final_alpha = self._restore_image_size(alpha, (original_w, original_h))

                if i >= len(committed_output) - overlap:
                    current_boundary_alphas.append(final_alpha.copy())
 
                mat_filename = os.path.join(core.matting_dir, f"{abs_frame:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, final_alpha)
 
                if abs_frame % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(abs_frame)
                        QApplication.processEvents()
                    except Exception:
                        pass

            prev_boundary_alphas = current_boundary_alphas if len(current_boundary_alphas) == overlap else None
 
            # Update progress
            pbar.update(1)
            batches_completed += 1
            current_progress = int(((operations_completed + batches_completed) * 100) / max(total_operations, 1))
            progress_dialog.setValue(min(current_progress, 99))
            QApplication.processEvents()
 
            core.DeviceManager.clear_cache()
 
        return batches_completed
