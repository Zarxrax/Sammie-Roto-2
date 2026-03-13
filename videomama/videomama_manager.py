# videomama/videomama_manager.py
"""
VideoMaMa Manager for Sammie-Roto 2

Manages the VideoMaMa (Mask-Guided Video Matting via Generative Prior) pipeline,
following the same interface pattern as MatAnyManager and RemovalManager.

VideoMaMa uses a diffusion-based approach (fine-tuned Stable Video Diffusion)
that processes batches of 16 frames at a time, unlike MatAnyone's frame-by-frame
temporal propagation. A windowing strategy with overlap and blending handles
videos longer than 16 frames.

Original: https://github.com/cvlab-kaist/VideoMaMa
License: CC BY-NC 4.0
"""

import os
import cv2
import shutil
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QProgressDialog, QApplication, QDialog, QVBoxLayout, QLabel,
    QProgressBar, QDialogButtonBox, QMessageBox
)

from sammie.settings_manager import get_settings_manager
from sammie.gui_widgets import show_message_dialog

# These are imported from sammie.sammie at runtime to avoid circular imports
# DeviceManager, VideoInfo, get_frame_extension, matting_dir, mask_dir, frames_dir


# .........................................................................................
# Constants
# .........................................................................................

VIDEOMAMA_BATCH_SIZE = 16  # Fixed by model architecture

# Resolution presets (width, height) - must be multiples of 64 for the VAE
# With CPU offload enabled, even 1024x576 fits in 24GB GPUs
VIDEOMAMA_RESOLUTION_MAP = {
    "512x288":   (512, 288),    # Fastest, lowest quality
    "768x448":   (768, 448),    # Balanced quality/speed
    "1024x576":  (1024, 576),   # Native model resolution, best quality
}
VIDEOMAMA_DEFAULT_RESOLUTION = "1024x576"


# .........................................................................................
# Download Worker and Dialog (runs snapshot_download in a background QThread)
# .........................................................................................

class _HFDownloadWorker(QObject):
    """
    Background worker that runs huggingface_hub.snapshot_download()
    in a separate thread to keep the UI responsive.
    """
    task_started = Signal(int, str)    # (index, description)
    task_done = Signal(int)            # (index,)
    task_error = Signal(int, str)      # (index, error_message)
    all_done = Signal()

    def __init__(self, tasks, parent=None):
        """
        Args:
            tasks: List of (repo_id, local_dir, description) tuples
        """
        super().__init__(parent)
        self._tasks = tasks
        self._abort = False

    @Slot()
    def run(self):
        from huggingface_hub import snapshot_download

        for idx, (repo_id, local_dir, description) in enumerate(self._tasks):
            if self._abort:
                return

            self.task_started.emit(idx, description)
            print(f"Downloading {repo_id} to {local_dir}...")

            try:
                # Only download component directories actually used by the pipeline.
                # The SVD repo includes ~18 GB of monolithic safetensors
                # (svd_xt.safetensors, svd_xt_image_decoder.safetensors) that
                # are NOT loaded — the pipeline uses the subfolder components.
                allow = [
                    "model_index.json",
                    "unet/**", "image_encoder/**", "vae/**",
                    "scheduler/**", "feature_extractor/**",
                    "*.pth",  # VideoMaMa UNet extras (dino_projection_mlp.pth)
                ]
                snapshot_download(
                    repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    allow_patterns=allow,
                )
                print(f"Downloaded {repo_id} successfully")
                self.task_done.emit(idx)
            except Exception as e:
                self.task_error.emit(idx, str(e))
                return

        self.all_done.emit()

    def abort(self):
        self._abort = True


class _VideoMaMaDownloadDialog(QDialog):
    """
    Modal dialog that downloads VideoMaMa models in a background thread.
    Follows the same pattern as Sammie's ModelDownloadDialog.
    """

    def __init__(self, tasks, parent=None):
        """
        Args:
            tasks: List of (repo_id, local_dir, description) tuples
        """
        super().__init__(parent)
        self._tasks = tasks
        self._success = False

        self.setWindowTitle("Downloading VideoMaMa Models")
        self.setMinimumWidth(480)
        self.setModal(True)

        self._build_ui()
        self._start_download()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self._status_label = QLabel("Preparing download...")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Indeterminate progress bar (snapshot_download doesn't give byte-level progress)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self._progress_bar)

        # Overall progress label
        self._overall_label = QLabel(f"0 / {len(self._tasks)} models")
        layout.addWidget(self._overall_label)

        # Error label (hidden until needed)
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: red;")
        self._error_label.hide()
        layout.addWidget(self._error_label)

        # Cancel / Close button
        self._button_box = QDialogButtonBox()
        self._cancel_btn = self._button_box.addButton(QDialogButtonBox.Cancel)
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._button_box)

    def _start_download(self):
        self._worker = _HFDownloadWorker(self._tasks)
        self._thread = QThread(self)

        self._worker.moveToThread(self._thread)

        # Wire signals
        self._thread.started.connect(self._worker.run)
        self._worker.task_started.connect(self._on_task_started)
        self._worker.task_done.connect(self._on_task_done)
        self._worker.task_error.connect(self._on_task_error)
        self._worker.all_done.connect(self._on_all_done)

        self._thread.start()

    @Slot(int, str)
    def _on_task_started(self, idx, description):
        self._status_label.setText(f"Downloading: {description}\nThis may take several minutes...")

    @Slot(int)
    def _on_task_done(self, idx):
        done = idx + 1
        self._overall_label.setText(f"{done} / {len(self._tasks)} models")

    @Slot(int, str)
    def _on_task_error(self, idx, message):
        self._thread.quit()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._status_label.setText(f"Download failed")
        self._error_label.setText(f"Error: {message}")
        self._error_label.show()
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)

    @Slot()
    def _on_all_done(self):
        self._thread.quit()
        self._success = True
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(100)
        self._status_label.setText("All VideoMaMa models downloaded successfully.")
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.accept)

    def _on_cancel(self):
        if hasattr(self, '_worker') and self._worker:
            self._worker.abort()
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Cancelling...")
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        self.reject()

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            if self._worker:
                self._worker.abort()
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)

    @property
    def succeeded(self):
        """True if all downloads completed successfully."""
        return self._success


class VideoMaMaManager:
    """
    Manager for VideoMaMa matting engine.
    
    Follows the same public interface as MatAnyManager:
        - load_model(parent_window) -> bool
        - unload_model()
        - run_matting(points_list, parent_window) -> int (1=success, 0=fail)
        - clear_matting()
        - propagated: bool
        - callbacks: list
        - add_callback(callback)
    
    Output is saved in the same format as MatAnyone:
        temp/matting/{frame:05d}/{object_id}.png (grayscale alpha matte)
    """

    def __init__(self):
        self.pipeline = None
        self.propagated = False
        self.callbacks = []

    def add_callback(self, callback):
        """Add callback for matting events"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")

    def load_model(self, parent_window=None):
        """
        Load the VideoMaMa pipeline (SVD base + fine-tuned UNet).
        
        Downloads models from Hugging Face if not present locally.
        Both repositories are public and do not require authentication.
        
        Args:
            parent_window: Parent window for download progress dialogs
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        from sammie.sammie import DeviceManager

        DeviceManager.clear_cache()
        device = DeviceManager.get_device()

        # Check for CUDA - VideoMaMa requires GPU
        if device.type == "cpu":
            print("VideoMaMa requires a CUDA GPU")
            return False

        # Download models if needed
        svd_path = os.path.join("checkpoints", "stable-video-diffusion-img2vid-xt")
        unet_path = os.path.join("checkpoints", "VideoMaMa")

        if not self._ensure_models_downloaded(svd_path, unet_path, parent_window):
            return False

        # Load the pipeline with ComfyUI-style memory optimizations
        try:
            from videomama.pipeline_svd_mask import VideoInferencePipeline

            # Read VAE tiling preference from settings
            settings_mgr = get_settings_manager()
            vae_tiling = settings_mgr.get_session_setting("videomama_vae_tiling", False)

            print("Loading VideoMaMa pipeline...")
            self.pipeline = VideoInferencePipeline(
                base_model_path=svd_path,
                unet_checkpoint_path=unet_path,
                weight_dtype=torch.float16,
                device=str(device),
                enable_model_cpu_offload=True,   # Move models to CPU when not in use
                vae_encode_chunk_size=4,          # Process VAE in small chunks
                attention_mode="auto",            # Use xformers if available, else SDPA
                enable_vae_tiling=vae_tiling,     # Tile-based VAE for lower VRAM peak
                enable_vae_slicing=True,          # Process VAE one image at a time
            )
            print("VideoMaMa pipeline loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading VideoMaMa pipeline: {e}")
            self.pipeline = None
            return False

    def _ensure_models_downloaded(self, svd_path, unet_path, parent_window=None):
        """
        Download VideoMaMa models from Hugging Face if not present.
        
        Uses huggingface_hub.snapshot_download() in a background QThread
        to keep the UI responsive. Both repos are public (no login required).
        
        Args:
            svd_path: Local path for SVD base model
            unet_path: Local path for VideoMaMa UNet checkpoint
            parent_window: Parent window for dialogs
            
        Returns:
            bool: True if models are ready, False if download failed/cancelled
        """
        svd_ready = os.path.exists(os.path.join(svd_path, "model_index.json"))
        unet_ready = os.path.exists(os.path.join(unet_path, "unet"))

        if svd_ready and unet_ready:
            return True

        # Try to import huggingface_hub
        try:
            from huggingface_hub import snapshot_download  # noqa: F401
        except ImportError:
            print("huggingface_hub is required for VideoMaMa model download.")
            print("Install with: pip install huggingface_hub")
            if parent_window:
                show_message_dialog(
                    parent_window,
                    title="Missing Dependency",
                    message="The 'huggingface_hub' package is required to download VideoMaMa models.\n\n"
                            "Install it with: pip install huggingface_hub",
                    type="warning"
                )
            return False

        # Confirm download with user (models are large)
        if parent_window:
            details = []
            total_gb = 0
            if not svd_ready:
                details.append("• SVD base model (~13 GB)\n"
                               "  UNet, VAE, Image Encoder, Scheduler")
                total_gb += 13
            if not unet_ready:
                details.append("• VideoMaMa fine-tuned UNet (~6 GB)")
                total_gb += 6

            details_text = "\n".join(details)
            reply = QMessageBox.question(
                parent_window,
                "Download VideoMaMa Models",
                f"VideoMaMa models need to be downloaded.\n\n"
                f"{details_text}\n\n"
                f"Total download: ~{total_gb} GB\n"
                f"No Hugging Face account is required.\n\n"
                f"Download now?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.No:
                return False

        # Build download task list
        tasks = []
        if not svd_ready:
            tasks.append((
                "stabilityai/stable-video-diffusion-img2vid-xt",
                svd_path,
                "SVD base model (~13 GB)"
            ))
        if not unet_ready:
            tasks.append((
                "SammyLim/VideoMaMa",
                unet_path,
                "VideoMaMa UNet (~6 GB)"
            ))

        # Run download in a background thread via modal dialog
        dialog = _VideoMaMaDownloadDialog(tasks, parent=parent_window)
        dialog.exec()
        return dialog.succeeded

    def unload_model(self):
        """Unload the VideoMaMa pipeline and free VRAM"""
        from sammie.sammie import DeviceManager

        self.pipeline = None
        DeviceManager.clear_cache()
        print("VideoMaMa pipeline unloaded")

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window):
        """
        Run VideoMaMa matting on all frames using a windowing strategy.
        
        VideoMaMa processes batches of 16 frames at a time. For videos longer
        than 16 frames, a sliding window with configurable overlap is used,
        and alpha mattes are blended at the overlap boundaries for smooth transitions.
        
        Args:
            points_list: List of point dictionaries (from Sammie's point manager)
            parent_window: Parent window for progress dialog
            
        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        from sammie.sammie import (
            DeviceManager, VideoInfo, get_frame_extension,
            matting_dir, mask_dir, frames_dir
        )

        if self.pipeline is None:
            print("VideoMaMa pipeline not loaded")
            return 0

        DeviceManager.clear_cache()
        frame_count = VideoInfo.total_frames

        # Get in/out points from settings
        settings_mgr = get_settings_manager()
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        overlap = settings_mgr.get_session_setting("videomama_overlap", 2)
        resolution = settings_mgr.get_session_setting(
            "videomama_resolution", VIDEOMAMA_DEFAULT_RESOLUTION
        )

        # Determine frame range to process
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1

        print(f"VideoMaMa: Processing frames {start_frame} to {end_frame} "
              f"({frames_to_process} frames, overlap={overlap}, resolution={resolution})")

        # Get unique object IDs from points list
        object_ids = sorted(list(set(
            point['object_id'] for point in points_list if 'object_id' in point
        )))
        if not object_ids:
            print("No objects found for matting")
            return 0

        # Load frame extension
        extension = get_frame_extension()

        # Create matting directory
        os.makedirs(matting_dir, exist_ok=True)

        # Calculate total operations for progress tracking
        total_batches = self._calculate_batch_count(frames_to_process, overlap)
        total_operations = total_batches * len(object_ids)

        # Create progress dialog
        progress_dialog = QProgressDialog(
            "Running VideoMaMa matting...", "Cancel", 0, 100, parent_window
        )
        progress_dialog.setWindowTitle("VideoMaMa Matting Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()

        pbar = tqdm(total=total_operations, desc="VideoMaMa Matting", unit="batch")
        operations_completed = 0

        # Process each object separately
        for object_id in object_ids:
            if progress_dialog.wasCanceled():
                break

            pbar.set_description(f"Object {object_id}")
            print(f"Processing object {object_id}...")

            success = self._process_object(
                object_id, start_frame, end_frame, overlap,
                extension, frames_dir, mask_dir, matting_dir,
                progress_dialog, pbar, operations_completed,
                total_operations, parent_window
            )

            if not success:
                break

            operations_completed += total_batches

        pbar.close()

        # Final cleanup
        DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("VideoMaMa matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(100)
            if frames_to_process == frame_count:
                self.propagated = True
            else:
                self.propagated = False
            print("VideoMaMa matting completed")
            self._notify('matting_complete')
            return 1

    def _process_object(self, object_id, start_frame, end_frame, overlap,
                        extension, frames_dir, mask_dir, matting_dir,
                        progress_dialog, pbar, operations_completed,
                        total_operations, parent_window):
        """
        Process a single object across all frames using windowed batching.
        
        Uses progressive save: frames that are not in overlap zones are saved
        and displayed immediately after each batch, giving the user live preview
        feedback (matching MatAnyone behavior). Overlap frames are saved once
        both contributing batches have been processed.
        
        Args:
            object_id: Object ID to process
            start_frame: First frame number (inclusive)
            end_frame: Last frame number (inclusive)
            overlap: Number of overlap frames between batches
            extension: Frame file extension (png, jpg, etc.)
            frames_dir: Directory containing source frames
            mask_dir: Directory containing segmentation masks
            matting_dir: Directory for output alpha mattes
            progress_dialog: Progress dialog
            pbar: tqdm progress bar
            operations_completed: Operations completed so far
            total_operations: Total operations
            parent_window: Parent window for display updates
            
        Returns:
            bool: True if successful, False if cancelled/failed
        """
        from sammie.sammie import DeviceManager

        frames_to_process = end_frame - start_frame + 1
        batch_size = VIDEOMAMA_BATCH_SIZE
        display_update_frequency = get_settings_manager().get_app_setting(
            "display_update_frequency", 5
        )

        # Get original frame dimensions for resizing output back
        first_frame_path = os.path.join(
            frames_dir, f"{start_frame:05d}.{extension}"
        )
        if os.path.exists(first_frame_path):
            first_frame = cv2.imread(first_frame_path)
            original_h, original_w = first_frame.shape[:2]
        else:
            # Fallback: use VideoInfo
            from sammie.sammie import VideoInfo
            original_w = VideoInfo.width
            original_h = VideoInfo.height

        # Generate batch windows
        windows = self._generate_windows(frames_to_process, batch_size, overlap)

        # Precompute which absolute frame numbers are in overlap zones
        # (these need contributions from 2 batches before they can be saved)
        overlap_frames = set()
        if overlap > 0:
            for batch_idx in range(len(windows) - 1):
                _, curr_end = windows[batch_idx]
                next_start, _ = windows[batch_idx + 1]
                # Frames in the overlap between batch_idx and batch_idx+1
                for f in range(start_frame + next_start, start_frame + curr_end):
                    overlap_frames.add(f)

        # Storage for alpha mattes pending overlap blending
        # Key: absolute frame number, Value: list of (alpha_array, weight) tuples
        alpha_accumulator = {}
        finalized_frames = set()  # Track frames already saved to disk

        for batch_idx, (window_start, window_end) in enumerate(windows):
            if progress_dialog.wasCanceled():
                return False

            # Convert window-relative indices to absolute frame numbers
            abs_start = start_frame + window_start
            abs_end = start_frame + window_end  # exclusive
            batch_length = abs_end - abs_start

            # Load frames and masks for this batch
            cond_frames, mask_frames, valid = self._load_batch(
                abs_start, abs_end, object_id, extension, frames_dir, mask_dir
            )

            if not valid:
                print(f"Warning: Skipping batch {batch_idx} for object {object_id} "
                      f"(frames {abs_start}-{abs_end - 1}) - missing data")
                if pbar is not None:
                    pbar.update(1)
                operations_completed += 1
                continue

            # Pad batch to 16 frames if needed (last batch may be shorter)
            cond_frames_padded, mask_frames_padded, pad_count = self._pad_batch(
                cond_frames, mask_frames, batch_size
            )

            # Run inference
            # Disable autocast because Sammie enables global bf16 autocast
            # for Ampere+ GPUs, which conflicts with the SVD pipeline's fp16 ops
            #
            # progress_callback keeps the UI responsive during the ~7s inference
            # by calling QApplication.processEvents() between pipeline stages
            # (CLIP encode, VAE encode, UNet, VAE decode)
            def _on_pipeline_progress(step, total, desc):
                progress_dialog.setLabelText(
                    f"Batch {batch_idx + 1}/{len(windows)} — {desc}"
                )
                QApplication.processEvents()

            try:
                with torch.amp.autocast('cuda', enabled=False):
                    output_frames = self.pipeline.run(
                        cond_frames=cond_frames_padded,
                        mask_frames=mask_frames_padded,
                        seed=42,
                        progress_callback=_on_pipeline_progress,
                    )
            except Exception as e:
                print(f"Error in VideoMaMa inference for batch {batch_idx}: {e}")
                raise

            # Extract alpha mattes (remove padding)
            actual_output = output_frames[:batch_length]

            # Add results to accumulator with blend weights
            # Skip frames that have already been finalized and saved
            for i, frame_pil in enumerate(actual_output):
                abs_frame = abs_start + i

                if abs_frame in finalized_frames:
                    continue  # Already saved from a previous batch

                # Convert PIL output to grayscale alpha matte
                alpha = self._extract_alpha_from_output(frame_pil)

                # Calculate blend weight for overlap regions
                weight = self._calculate_blend_weight(
                    i, batch_length, window_start, window_end,
                    windows, batch_idx, overlap
                )

                if abs_frame not in alpha_accumulator:
                    alpha_accumulator[abs_frame] = []
                alpha_accumulator[abs_frame].append((alpha, weight))

            # --- Progressive save: finalize and display completed frames ---
            # Non-overlap frames can be saved immediately (1 contribution).
            # Overlap frames need 2 contributions before they can be blended and saved.
            frames_to_finalize = []
            for abs_frame in sorted(alpha_accumulator.keys()):
                entries = alpha_accumulator[abs_frame]

                if abs_frame not in overlap_frames:
                    # Non-overlap frame: ready after 1 contribution
                    frames_to_finalize.append(abs_frame)
                elif len(entries) >= 2:
                    # Overlap frame: ready after both batches contributed
                    frames_to_finalize.append(abs_frame)

            # Save finalized frames and update display
            for abs_frame in frames_to_finalize:
                alpha_weight_list = alpha_accumulator.pop(abs_frame)
                final_alpha = self._finalize_alpha(
                    alpha_weight_list, original_w, original_h
                )

                # Save in MatAnyone-compatible format
                mat_filename = os.path.join(
                    matting_dir, f"{abs_frame:05d}", f"{object_id}.png"
                )
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, final_alpha)
                finalized_frames.add(abs_frame)

                # Live display update (same pattern as MatAnyone)
                if abs_frame % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(abs_frame)
                        QApplication.processEvents()
                    except Exception:
                        pass

            # Update progress
            if pbar is not None:
                pbar.update(1)
            operations_completed += 1
            current_progress = int((operations_completed * 100) / max(total_operations, 1))
            progress_dialog.setValue(min(current_progress, 99))
            QApplication.processEvents()

            DeviceManager.clear_cache()

        # Save any remaining frames in the accumulator (last overlap frames)
        for abs_frame in sorted(alpha_accumulator.keys()):
            alpha_weight_list = alpha_accumulator[abs_frame]
            final_alpha = self._finalize_alpha(
                alpha_weight_list, original_w, original_h
            )

            mat_filename = os.path.join(
                matting_dir, f"{abs_frame:05d}", f"{object_id}.png"
            )
            os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
            cv2.imwrite(mat_filename, final_alpha)

            if abs_frame % display_update_frequency == 0:
                try:
                    parent_window.frame_slider.setValue(abs_frame)
                    QApplication.processEvents()
                except Exception:
                    pass

        return True

    def _finalize_alpha(self, alpha_weight_list, original_w, original_h):
        """
        Blend and resize an alpha matte to its final output form.
        
        Handles both single-contribution frames (no blending) and
        multi-contribution overlap frames (weighted average).
        
        Args:
            alpha_weight_list: List of (alpha_array, weight) tuples
            original_w: Original video width
            original_h: Original video height
            
        Returns:
            numpy array: Final alpha matte at original resolution, uint8
        """
        if len(alpha_weight_list) == 1:
            final_alpha = alpha_weight_list[0][0]
        else:
            total_weight = sum(w for _, w in alpha_weight_list)
            blended = np.zeros_like(alpha_weight_list[0][0], dtype=np.float32)
            for alpha, weight in alpha_weight_list:
                blended += alpha.astype(np.float32) * (weight / total_weight)
            final_alpha = np.clip(blended, 0, 255).astype(np.uint8)

        # Resize to original video resolution using PIL LANCZOS
        if final_alpha.shape[0] != original_h or final_alpha.shape[1] != original_w:
            alpha_pil = Image.fromarray(final_alpha, mode='L')
            alpha_pil = alpha_pil.resize((original_w, original_h), Image.LANCZOS)
            final_alpha = np.array(alpha_pil)

        return final_alpha

    def _load_batch(self, abs_start, abs_end, object_id, extension,
                    frames_dir, mask_dir):
        """
        Load a batch of frames and corresponding masks from disk.
        
        Args:
            abs_start: Absolute start frame (inclusive)
            abs_end: Absolute end frame (exclusive)
            object_id: Object ID for mask lookup
            extension: Frame file extension
            frames_dir: Directory containing source frames
            mask_dir: Directory containing segmentation masks
            
        Returns:
            tuple: (cond_frames_pil, mask_frames_pil, valid)
        """
        # Get internal resolution from settings
        settings_mgr = get_settings_manager()
        res_key = settings_mgr.get_session_setting(
            "videomama_resolution", VIDEOMAMA_DEFAULT_RESOLUTION
        )
        target_w, target_h = VIDEOMAMA_RESOLUTION_MAP.get(
            res_key, VIDEOMAMA_RESOLUTION_MAP[VIDEOMAMA_DEFAULT_RESOLUTION]
        )

        cond_frames = []
        mask_frames = []

        for frame_num in range(abs_start, abs_end):
            # Load source frame
            frame_path = os.path.join(frames_dir, f"{frame_num:05d}.{extension}")
            if not os.path.exists(frame_path):
                print(f"Warning: Frame not found: {frame_path}")
                return [], [], False

            frame_img = Image.open(frame_path).convert("RGB")

            # Load segmentation mask
            mask_path = os.path.join(mask_dir, f"{frame_num:05d}", f"{object_id}.png")
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert("L")
            else:
                # If no mask exists for this frame, use a blank mask
                mask_img = Image.new("L", frame_img.size, 0)

            # Resize to configured internal resolution (LANCZOS for best quality)
            frame_resized = frame_img.resize(
                (target_w, target_h), Image.LANCZOS
            )
            mask_resized = mask_img.resize(
                (target_w, target_h), Image.LANCZOS
            )

            # Binarize mask
            mask_resized = mask_resized.point(
                lambda p: 255 if p > 127 else 0, mode='L'
            )

            cond_frames.append(frame_resized)
            mask_frames.append(mask_resized)

        return cond_frames, mask_frames, True

    def _pad_batch(self, cond_frames, mask_frames, target_size):
        """
        Pad a batch to the target size by repeating the last frame.
        
        The VideoMaMa model expects exactly 16 frames. If the last batch
        has fewer frames, we pad by repeating the last frame/mask pair.
        
        Args:
            cond_frames: List of PIL conditioning frames
            mask_frames: List of PIL mask frames
            target_size: Target batch size (16)
            
        Returns:
            tuple: (padded_cond, padded_mask, pad_count)
        """
        current_size = len(cond_frames)
        if current_size >= target_size:
            return cond_frames[:target_size], mask_frames[:target_size], 0

        pad_count = target_size - current_size
        last_frame = cond_frames[-1]
        last_mask = mask_frames[-1]

        padded_cond = cond_frames + [last_frame] * pad_count
        padded_mask = mask_frames + [last_mask] * pad_count

        return padded_cond, padded_mask, pad_count

    def _extract_alpha_from_output(self, frame_pil):
        """
        Extract a grayscale alpha matte from the VideoMaMa output frame.
        
        Uses cv2.cvtColor with perceptual luminance weights (0.299R + 0.587G + 0.114B)
        matching the official VideoMaMa demo, which produces slightly better edge
        contrast than a simple RGB mean.
        
        Args:
            frame_pil: Output PIL image from the pipeline
            
        Returns:
            numpy array: Grayscale alpha matte (H, W), uint8, 0-255
        """
        frame_np = np.array(frame_pil)

        if frame_np.ndim == 3:
            # Perceptual luminance (same as official VideoMaMa demo)
            alpha = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        else:
            alpha = frame_np

        return alpha.astype(np.uint8)

    def _generate_windows(self, total_frames, batch_size, overlap):
        """
        Generate sliding window positions for batched processing.
        
        Args:
            total_frames: Total number of frames to process
            batch_size: Frames per batch (16)
            overlap: Number of overlapping frames between batches
            
        Returns:
            list of (start, end) tuples (end is exclusive)
        """
        if total_frames <= batch_size:
            return [(0, total_frames)]

        windows = []
        step = batch_size - overlap
        pos = 0

        while pos < total_frames:
            end = min(pos + batch_size, total_frames)
            windows.append((pos, end))

            if end >= total_frames:
                break
            pos += step

        return windows

    def _calculate_batch_count(self, total_frames, overlap):
        """Calculate the number of batches needed for the given frame count"""
        if total_frames <= VIDEOMAMA_BATCH_SIZE:
            return 1
        step = VIDEOMAMA_BATCH_SIZE - overlap
        return max(1, (total_frames - overlap + step - 1) // step)

    def _calculate_blend_weight(self, local_idx, batch_length, window_start,
                                window_end, windows, batch_idx, overlap):
        """
        Calculate the blending weight for a frame in an overlap region.
        
        For non-overlap frames, weight is 1.0.
        For overlap frames, weight linearly ramps from 0.0 to 1.0 (for the
        newer batch) or 1.0 to 0.0 (for the older batch).
        
        Args:
            local_idx: Frame index within current batch
            batch_length: Actual length of current batch (may be < 16)
            window_start: Start of current window (relative to processing range)
            window_end: End of current window (exclusive)
            windows: List of all window (start, end) tuples
            batch_idx: Index of current batch
            overlap: Number of overlap frames
            
        Returns:
            float: Blend weight (0.0 to 1.0)
        """
        if overlap == 0:
            return 1.0

        is_first_batch = (batch_idx == 0)
        is_last_batch = (batch_idx == len(windows) - 1)

        # Frames at the end of a non-last batch that overlap with the next batch
        if not is_last_batch and local_idx >= (batch_length - overlap):
            # Ramp down: weight decreases from 1.0 to 0.0
            overlap_pos = local_idx - (batch_length - overlap)
            return 1.0 - (overlap_pos / overlap)

        # Frames at the start of a non-first batch that overlap with the previous batch
        if not is_first_batch and local_idx < overlap:
            # Ramp up: weight increases from 0.0 to 1.0
            return local_idx / overlap

        return 1.0

    def clear_matting(self):
        """Clear matting data"""
        from sammie.sammie import matting_dir

        if os.path.exists(matting_dir):
            shutil.rmtree(matting_dir)
        os.makedirs(matting_dir)
        self.propagated = False
        print("Matting data cleared")
