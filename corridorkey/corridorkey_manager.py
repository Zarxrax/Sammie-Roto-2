"""
CorridorKey Manager for Sammie-Roto 2.

Follows the same interface pattern as MatAnyManager and VideoMaMaManager:
load_model → run → unload_model, with HuggingFace auto-download and
frame-by-frame live preview.
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QDialog, QProgressBar, QProgressDialog,
    QPushButton, QVBoxLayout, QLabel,
)
from PySide6.QtCore import Qt, QObject, QThread, Signal
from tqdm import tqdm

from sammie.settings_manager import get_settings_manager

# HuggingFace model info
CORRIDORKEY_HF_REPO = "nikopueringer/CorridorKey_v1.0"
CORRIDORKEY_HF_FILE = "CorridorKey_v1.0.pth"
CORRIDORKEY_CHECKPOINT_DIR = os.path.join("checkpoints", "CorridorKey")
CORRIDORKEY_CHECKPOINT_FILE = os.path.join(CORRIDORKEY_CHECKPOINT_DIR, CORRIDORKEY_HF_FILE)


# ==================== Download Worker ====================

class _HFDownloadWorker(QObject):
    """Worker thread for downloading CorridorKey checkpoint from HuggingFace."""
    finished = Signal(bool, str)
    progress = Signal(str)

    def __init__(self, repo_id, filename, local_dir):
        super().__init__()
        self.repo_id = repo_id
        self.filename = filename
        self.local_dir = local_dir

    def run(self):
        try:
            from huggingface_hub import hf_hub_download
            self.progress.emit(f"Downloading {self.filename}...")
            hf_hub_download(
                repo_id=self.repo_id,
                filename=self.filename,
                local_dir=self.local_dir,
            )
            self.finished.emit(True, "Download complete")
        except Exception as e:
            self.finished.emit(False, str(e))


class _CorridorKeyDownloadDialog(QDialog):
    """Download dialog for CorridorKey checkpoint (~383 MB)."""

    def __init__(self, repo_id, filename, local_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading CorridorKey Model")
        self.setFixedWidth(450)
        self.setModal(True)
        self._success = False

        layout = QVBoxLayout(self)
        self.status_label = QLabel(f"Downloading CorridorKey model (~383 MB)...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        layout.addWidget(self.progress_bar)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)

        # Start download in background thread
        self._thread = QThread()
        self._worker = _HFDownloadWorker(repo_id, filename, local_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, msg):
        self.status_label.setText(msg)

    def _on_finished(self, success, msg):
        self._success = success
        self._thread.quit()
        self._thread.wait()
        if success:
            self.accept()
        else:
            self.status_label.setText(f"Download failed: {msg}")
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)

    def was_successful(self):
        return self._success


# ==================== Manager ====================

class CorridorKeyManager:
    """
    Manager for CorridorKey AI green screen keying.

    Interface matches MatAnyManager/VideoMaMaManager:
    - load_model(parent_window) → bool
    - run_corridorkey(points_list, parent_window) → int (1=success, 0=fail)
    - unload_model()
    - clear_corridorkey()
    """

    def __init__(self):
        self.engine = None
        self.propagated = False
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        for cb in self.callbacks:
            try:
                cb(action, **kwargs)
            except Exception as e:
                print(f"CorridorKey callback error: {e}")

    # -------------------- Model Lifecycle --------------------

    def load_model(self, parent_window=None):
        """
        Load the CorridorKey engine.

        Downloads the checkpoint from HuggingFace if not present (~383 MB).
        """
        from sammie.sammie import DeviceManager

        DeviceManager.clear_cache()
        device = DeviceManager.get_device()

        if device.type == "cpu":
            print("CorridorKey requires a CUDA GPU")
            return False

        # Download checkpoint if needed
        if not self._ensure_model_downloaded(parent_window):
            return False

        try:
            from corridorkey.inference_engine import CorridorKeyEngine

            settings_mgr = get_settings_manager()

            checkpoint = getattr(self, '_checkpoint_override', CORRIDORKEY_CHECKPOINT_FILE)

            quality = settings_mgr.get_session_setting("corridorkey_quality", "auto")
            quality_to_imgsize = {"low": 1024, "medium": 1536, "high": 2048}
            if quality in quality_to_imgsize:
                img_size = quality_to_imgsize[quality]
            else:
                try:
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    if vram_gb >= 14:
                        img_size = 2048
                    elif vram_gb >= 10:
                        img_size = 1536
                    else:
                        img_size = 1024
                except Exception:
                    img_size = 1024

            use_tiling = settings_mgr.get_session_setting("corridorkey_tiling", False)
            print(f"Loading CorridorKey engine (checkpoint: {checkpoint}, "
                  f"img_size: {img_size}, quality: {quality}, tiling: {use_tiling})...")
            self.engine = CorridorKeyEngine(
                checkpoint_path=checkpoint,
                device=str(device),
                img_size=img_size,
                use_refiner=True,
                optimization_mode='auto',
                force_tiling=use_tiling,
            )
            print("CorridorKey engine loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading CorridorKey engine: {e}")
            self.engine = None
            return False

    def unload_model(self):
        """Unload the CorridorKey engine and free VRAM."""
        if self.engine is not None:
            self.engine.unload()
            self.engine = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("CorridorKey engine unloaded")

    def _ensure_model_downloaded(self, parent_window=None):
        """Download the CorridorKey checkpoint if not present."""
        os.makedirs(CORRIDORKEY_CHECKPOINT_DIR, exist_ok=True)

        if os.path.exists(CORRIDORKEY_CHECKPOINT_FILE):
            return True

        # Check for any .pth in the directory (different name)
        for f in os.listdir(CORRIDORKEY_CHECKPOINT_DIR):
            if f.endswith('.pth'):
                # Use whatever .pth is there
                self._checkpoint_override = os.path.join(CORRIDORKEY_CHECKPOINT_DIR, f)
                return True

        print("CorridorKey checkpoint not found, downloading...")
        dialog = _CorridorKeyDownloadDialog(
            CORRIDORKEY_HF_REPO, CORRIDORKEY_HF_FILE,
            CORRIDORKEY_CHECKPOINT_DIR, parent_window
        )
        dialog.exec()
        return dialog.was_successful()

    # -------------------- Processing --------------------

    def run_corridorkey(self, points_list, parent_window=None):
        """
        Run CorridorKey on all frames for all tracked objects.

        Uses the coarse mask (from segmentation or matting) as guide input.
        Processes frame-by-frame with live display preview.

        Args:
            points_list: List of point dicts (from PointManager)
            parent_window: MainWindow for display updates

        Returns:
            int: 1 if successful, 0 if failed/cancelled
        """
        from sammie.sammie import DeviceManager, VideoInfo
        from sammie.settings_manager import get_settings_manager, get_frame_extension

        settings_mgr = get_settings_manager()

        # Read settings
        mask_source = settings_mgr.get_session_setting("corridorkey_mask_source", "Segmentation")
        refiner_scale = settings_mgr.get_session_setting("corridorkey_refiner_scale", 1.0)
        despill_strength = settings_mgr.get_session_setting("corridorkey_despill", 1.0)
        auto_despeckle = settings_mgr.get_session_setting("corridorkey_despeckle", True)
        despeckle_size = settings_mgr.get_session_setting("corridorkey_despeckle_size", 400)

        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        display_update_freq = settings_mgr.get_app_setting("display_update_frequency", 5)

        frame_count = VideoInfo.total_frames
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1

        # Get unique object IDs
        object_ids = sorted(set(
            p['object_id'] for p in points_list if 'object_id' in p
        ))
        if not object_ids:
            print("No objects found for CorridorKey")
            return 0

        extension = get_frame_extension()
        session_dir = settings_mgr.get_session_dir()
        frames_dir = os.path.join(session_dir, "frames")

        # Determine mask source directory
        if mask_source == "Matting":
            mask_dir = os.path.join(session_dir, "matting")
        else:
            mask_dir = os.path.join(session_dir, "masks")

        # Output directories
        corridorkey_dir = os.path.join(session_dir, "corridorkey")
        alpha_dir = os.path.join(corridorkey_dir, "alpha")
        fg_dir = os.path.join(corridorkey_dir, "fg")
        comp_dir = os.path.join(corridorkey_dir, "comp")

        # Also save alpha to matting dir for viewer compatibility
        matting_dir = os.path.join(session_dir, "matting")

        total_ops = frames_to_process * len(object_ids)

        print(f"CorridorKey: Processing frames {start_frame}-{end_frame} "
              f"({frames_to_process} frames, {len(object_ids)} objects, "
              f"mask_source={mask_source})")

        # Progress dialog
        progress_dialog = QProgressDialog(
            "Running CorridorKey...", "Cancel", 0, 100, parent_window
        )
        progress_dialog.setWindowTitle("CorridorKey Matting Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setValue(0)
        QApplication.processEvents()

        # Load model
        progress_dialog.setLabelText("Loading CorridorKey model...")
        QApplication.processEvents()

        if self.engine is None:
            if not self.load_model(parent_window):
                progress_dialog.close()
                return 0

        try:
            ops_done = 0

            for object_id in object_ids:
                pbar = tqdm(total=frames_to_process, desc=f"Object {object_id}",
                            unit="frame")

                for frame_num in range(start_frame, end_frame + 1):
                    if progress_dialog.wasCanceled():
                        pbar.close()
                        self.unload_model()
                        return 0

                    # Load RGB frame
                    frame_path = os.path.join(
                        frames_dir, f"{frame_num:05d}.{extension}"
                    )
                    if not os.path.exists(frame_path):
                        print(f"Warning: Frame not found: {frame_path}")
                        pbar.update(1)
                        ops_done += 1
                        continue

                    frame_bgr = cv2.imread(frame_path)
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                    # Load coarse mask
                    mask_path = os.path.join(
                        mask_dir, f"{frame_num:05d}", f"{object_id}.png"
                    )
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask_source != "Matting":
                            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    else:
                        # No mask for this frame — skip
                        pbar.update(1)
                        ops_done += 1
                        continue

                    # Run CorridorKey inference
                    result = self.engine.process_frame(
                        frame_rgb,
                        mask,
                        refiner_scale=refiner_scale,
                        despill_strength=despill_strength,
                        auto_despeckle=auto_despeckle,
                        despeckle_size=despeckle_size,
                        input_is_linear=False,  # Sammie frames are sRGB PNG
                    )

                    # Save all outputs
                    self._save_results(
                        result, frame_num, object_id,
                        alpha_dir, fg_dir, comp_dir, matting_dir
                    )

                    # Free VRAM between frames (same pattern as MatAnyone)
                    torch.cuda.empty_cache()

                    # Live display update
                    if frame_num % display_update_freq == 0:
                        try:
                            parent_window.frame_slider.setValue(frame_num)
                        except Exception:
                            pass

                    # Progress
                    pbar.update(1)
                    ops_done += 1
                    progress = int((ops_done * 100) / max(total_ops, 1))
                    progress_dialog.setValue(min(progress, 99))
                    progress_dialog.setLabelText(
                        f"CorridorKey — Object {object_id}, "
                        f"Frame {frame_num}/{end_frame}"
                    )
                    QApplication.processEvents()

                pbar.close()

            # Done
            progress_dialog.setValue(100)
            progress_dialog.close()

            self.propagated = True
            print("CorridorKey processing completed")
            self._notify('corridorkey_complete')

            # Unload to free VRAM for other operations
            self.unload_model()
            return 1

        except Exception as e:
            print(f"CorridorKey error: {e}")
            progress_dialog.close()
            self.unload_model()
            return 0

    def _save_results(self, result, frame_num, object_id,
                      alpha_dir, fg_dir, comp_dir, matting_dir):
        """
        Save all CorridorKey outputs for a single frame as 16-bit PNGs.

        Saves:
        - alpha as 16-bit grayscale PNG in corridorkey/alpha/
        - alpha as 8-bit grayscale PNG in matting/ (viewer compatibility)
        - fg as 16-bit RGB PNG in corridorkey/fg/
        - comp as 16-bit RGB PNG in corridorkey/comp/
        """
        frame_str = f"{frame_num:05d}"

        alpha = result['alpha']
        if alpha.ndim == 3:
            alpha = alpha[:, :, 0]
        alpha_clamped = np.clip(alpha, 0, 1)

        # 16-bit alpha for CK viewer/export
        alpha_u16 = (alpha_clamped * 65535).astype(np.uint16)
        alpha_out = os.path.join(alpha_dir, frame_str, f"{object_id}.png")
        os.makedirs(os.path.dirname(alpha_out), exist_ok=True)
        cv2.imwrite(alpha_out, alpha_u16)

        # 8-bit copy for matting viewer compatibility
        alpha_u8 = (alpha_clamped * 255).astype(np.uint8)
        mat_out = os.path.join(matting_dir, frame_str, f"{object_id}.png")
        os.makedirs(os.path.dirname(mat_out), exist_ok=True)
        cv2.imwrite(mat_out, alpha_u8)

        # 16-bit FG
        fg = result['fg']
        fg_u16 = (np.clip(fg, 0, 1) * 65535).astype(np.uint16)
        fg_bgr = cv2.cvtColor(fg_u16, cv2.COLOR_RGB2BGR)
        fg_out = os.path.join(fg_dir, frame_str, f"{object_id}.png")
        os.makedirs(os.path.dirname(fg_out), exist_ok=True)
        cv2.imwrite(fg_out, fg_bgr)

        # 16-bit Comp
        comp = result['comp']
        comp_u16 = (np.clip(comp, 0, 1) * 65535).astype(np.uint16)
        comp_bgr = cv2.cvtColor(comp_u16, cv2.COLOR_RGB2BGR)
        comp_out = os.path.join(comp_dir, frame_str, f"{object_id}.png")
        os.makedirs(os.path.dirname(comp_out), exist_ok=True)
        cv2.imwrite(comp_out, comp_bgr)

    def clear_corridorkey(self):
        """Clear CorridorKey results."""
        self.propagated = False
        self._notify('corridorkey_cleared')
        print("CorridorKey data cleared")
