# sammie/removal.py
import cv2
import os
import numpy as np
import torch
import shutil
import gc
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt
from sammie import core
from sammie.settings_manager import get_settings_manager
from sammie.model_downloader import ensure_models


class RemovalManager(core.CallbackMixin):
    """Manager for object removal operations"""

    def __init__(self):
        super().__init__()
        self.pipe = None
        self.propainterx_pipeline = None
        self.propagated = False  # whether removal has been completed

    def load_minimax_model(self, parent_window=None):
        from diffusers.models import AutoencoderKLWan
        from diffusers.schedulers import UniPCMultistepScheduler
        from minimax_remover.pipeline_minimax_remover import Minimax_Remover_Pipeline
        from minimax_remover.transformer_minimax_remover import Transformer3DModel

        settings_mgr = get_settings_manager()
        minimax_vae_tiling = settings_mgr.get_session_setting("minimax_vae_tiling", False)
        core.DeviceManager.clear_cache()
        device = core.DeviceManager.get_device()

        # Move the models into minimax folder (for better organization) if they aren't already there
        # This will be removed in a future version
        try:
            if os.path.exists("checkpoints/transformer"):
                shutil.copytree("checkpoints/transformer","checkpoints/minimax/transformer",dirs_exist_ok=True)
                shutil.rmtree("checkpoints/transformer")
            if os.path.exists("checkpoints/vae"):
                shutil.copytree("checkpoints/vae","checkpoints/minimax/vae",dirs_exist_ok=True)
                shutil.rmtree("checkpoints/vae")
            if os.path.exists("checkpoints/scheduler"):
                shutil.copytree("checkpoints/scheduler","checkpoints/minimax/scheduler",dirs_exist_ok=True)
                shutil.rmtree("checkpoints/scheduler")
        except Exception as e:
            print(f"Warning: Failed to move existing model files: {e}")


        # download models if they don't exist
        if not ensure_models(["minimax_transformer", "minimax_vae"], parent=parent_window):
            return False

        # Load the minimax remover models
        model_path = "./checkpoints/minimax/"
        vae = AutoencoderKLWan.from_pretrained(
            f"{model_path}/vae",
            torch_dtype=torch.float16,
            device=device
        )
        transformer = Transformer3DModel.from_pretrained(
            f"{model_path}/transformer",
            torch_dtype=torch.float16,
            device=device
        )
        scheduler = UniPCMultistepScheduler.from_pretrained(
            f"{model_path}/scheduler"
        )

        self.pipe = Minimax_Remover_Pipeline(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler
        )

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        if minimax_vae_tiling:
            self.pipe.enable_vae_tiling()
            print("VAE tiling enabled")

        return self.pipe

    def unload_minimax_model(self):
        if self.pipe is not None:
            try:
                self.pipe.maybe_free_model_hooks()
            except Exception as e:
                print(f"Warning: Could not free model hooks: {e}")

            self.pipe.transformer = None
            self.pipe.vae = None
            self.pipe.scheduler = None
            self.pipe = None

        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded MiniMax-Remover model")

    def run_object_removal_minimax(self, points, parent_window=None):
        """
        Run MiniMax-Remover (inpainting) on all frames.
        Combines masks for all objects and loads all frames and masks into memory upfront.

        Args:
            points (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        frame_count = core.VideoInfo.total_frames
        settings_mgr = get_settings_manager()
        device = core.DeviceManager.get_device()
        self.propagated = False

        # Determine frame range to process
        start_frame, end_frame, frames_to_process = core.get_frame_range()

        print(f"Processing removal from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get settings
        minimax_steps = settings_mgr.get_session_setting("minimax_steps", 6)
        inpaint_grow = settings_mgr.get_session_setting("inpaint_grow", 0)

        # Pass in a blank image to see what it gets resized to
        blank_image = np.zeros((core.VideoInfo.height, core.VideoInfo.width, 1), dtype=np.uint8)
        blank_image = self.resize_image_minimax(blank_image, mask=True)
        resized_h, resized_w = blank_image.shape[:2]

        # Create progress dialog
        progress_dialog = QProgressDialog("Loading MiniMax-Remover model...", "Cancel", 0, 0, parent_window)
        progress_dialog.setWindowTitle("Object Removal Progress")
        progress_dialog.setWindowModality(Qt.ApplicationModal)
        progress_dialog.show()
        print(f"Loading MiniMax-Remover model to {device} with resolution {resized_w}x{resized_h}...")
        QApplication.processEvents()

        # Load model
        self.load_minimax_model(parent_window=parent_window)
        if self.pipe is None:
            print("Error loading MiniMax-Remover model")
            progress_dialog.close()
            return 0

        # Link progress dialog to pipeline
        self.pipe.progress_dialog = progress_dialog

        # Create output directory if it doesn't exist (don't clear existing frames)
        os.makedirs(core.removal_dir, exist_ok=True)

        # Load and prepare data
        print("Loading frames and masks...")
        QApplication.processEvents()
        frames, masks = self._load_all_frames_and_masks(points, inpaint_grow=inpaint_grow,
                                                        start_frame=start_frame, end_frame=end_frame)

        # Pad frames
        pad_frames = (4 - (frames_to_process % 4)) % 4 + 1
        if pad_frames > 0:
            for _ in range(pad_frames):
                frames.append(frames[-1].copy())
                masks.append(masks[-1].copy())

        # Convert to tensors
        device = core.DeviceManager.get_device()
        frames = torch.from_numpy(np.stack(frames)).half().to(device)
        masks = torch.from_numpy(np.stack(masks)).half().to(device)
        masks = masks[:, :, :, None]

        # Run inference
        print("Running inference...")
        QApplication.processEvents()
        device_type = core.DeviceManager.get_device().type
        try:
            with torch.autocast(device_type=device_type, enabled=False), torch.no_grad():
                output = self.pipe(
                    images=frames,
                    masks=masks,
                    num_frames=masks.shape[0],
                    height=masks.shape[1],
                    width=masks.shape[2],
                    num_inference_steps=minimax_steps,
                    generator=torch.Generator(device=device).manual_seed(42),
                ).frames[0]
        except RuntimeError as e:
            self.propagated = False
            if "cancelled" in str(e).lower():
                print("User cancelled MiniMax processing.")
                progress_dialog.close()
                return 0
            else:
                progress_dialog.close()
                raise
        finally:
            if frames is not None:
                del frames
            if masks is not None:
                del masks

        # Remove padding and convert
        if pad_frames > 0:
            output = output[:frames_to_process]
        output = np.uint8(output * 255)

        # Save frames
        print("Saving frames...")
        progress_dialog.setLabelText("Saving frames...")
        QApplication.processEvents()
        extension = core.get_frame_extension()

        # Save processed frames
        for i, frame in enumerate(output):
            frame_number = start_frame + i
            composited = self.composite_removal_over_original(frame, frame_number, points)
            output_path = os.path.join(core.removal_dir, f"{frame_number:05d}.{extension}")
            frame_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, frame_bgr)

        if frame_count == frames_to_process:  # only set propagated if the whole video was processed
            self.propagated = True
        else:
            self.propagated = False
        print("Processing complete!")
        progress_dialog.close()
        return True

    def _load_all_frames_and_masks(self, points_list, inpaint_grow=5, start_frame=0, end_frame=None):
        """
        Load all frames and corresponding combined masks into memory, while resizing and processing.
        Used for MiniMax-Remover object removal.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame info.
            inpaint_grow (int): Optional grow/shrink parameter for masks.
            start_frame (int): Starting frame (inclusive)
            end_frame (int): Ending frame (inclusive). If None, process to end of video.

        Returns:
            tuple: (frames, masks)
                - frames: list of np.ndarray (BGR images)
                - masks: list of np.ndarray (uint8, single-channel)
        """
        frame_count = core.VideoInfo.total_frames
        if end_frame is None:
            end_frame = frame_count - 1
        extension = core.get_frame_extension()

        # Get all unique object IDs
        object_ids = sorted(list(set(p['object_id'] for p in points_list if 'object_id' in p)))
        if not object_ids:
            print("No objects found — returning empty frame/mask arrays.")
            return [], []

        frames = []
        masks = []

        for frame_number in range(start_frame, end_frame + 1):
            frame_path = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")

            # Load frame
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Combine masks for all objects on this frame (falls back to an
            # all-black mask, matching the frame's shape, if none are found)
            combined_mask, _ = core.load_combined_mask(frame_number, object_ids)
            if combined_mask is None:
                combined_mask = np.zeros(frame.shape[:2], np.uint8)

            # Apply segmentation postprocessing (holes, dots, border_fix, grow)
            combined_mask = core.apply_mask_postprocessing(combined_mask)

            # Only apply the inpaint_grow portion (grow was already applied in core.apply_mask_postprocessing)
            if inpaint_grow != 0:
                combined_mask = core.grow_shrink(combined_mask, inpaint_grow)

            # Resize and normalize frame and mask
            frame = self.resize_image_minimax(frame)
            frame = frame.astype(np.float32) / 127.5 - 1.0
            combined_mask = self.resize_image_minimax(combined_mask, mask=True)
            combined_mask = (combined_mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

            frames.append(frame)
            masks.append(combined_mask)

        print(f"Loaded {len(frames)} frames and masks into memory.")
        return frames, masks

    def composite_removal_over_original(self, processed_frame, frame_number, points):
        """
        Composite a processed removal frame over the original full-resolution frame.
        Only the masked regions use the processed (lower-res) frame, everything else
        uses the original full-resolution frame.

        Args:
            processed_frame: Processed frame from removal pipeline (RGB, may be lower resolution)
            frame_number: Frame number to load original for
            points: Points list to determine which masks to use

        Returns:
            np.ndarray: Composited RGB frame at original resolution
        """
        settings_mgr = get_settings_manager()
        original_size = (core.VideoInfo.width, core.VideoInfo.height)  # (width, height) for cv2.resize

        # Load original full-resolution image
        original_frame = core.load_base_frame(frame_number)
        if original_frame is None:
            print(f"Warning: Could not load original frame {frame_number}, using processed frame only")
            return cv2.resize(processed_frame, original_size, interpolation=cv2.INTER_LINEAR)

        # Resize processed frame to original size
        frame_restored = cv2.resize(processed_frame, original_size, interpolation=cv2.INTER_LINEAR)

        # Load original masks
        original_mask = core.load_masks_for_frame(frame_number, points, return_combined=True)

        if original_mask is None:
            return original_frame

        # Apply segmentation postprocessing (holes, dots, border_fix, grow)
        original_mask = core.apply_mask_postprocessing(original_mask)

        # Apply additional inpaint_grow
        inpaint_grow = settings_mgr.get_session_setting("inpaint_grow", 0)
        inpaint_grow = inpaint_grow + 21  # make it much larger to account for feathering

        if inpaint_grow != 0:
            original_mask = core.grow_shrink(original_mask, inpaint_grow)

        # Apply feathering to the mask for smoother transitions
        feather_radius = 10
        mask_feathered = cv2.GaussianBlur(original_mask, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)

        # Convert mask to 3-channel and normalize to [0, 1]
        mask_3channel = np.stack([mask_feathered] * 3, axis=-1).astype(np.float32) / 255.0

        # Composite in floating point to avoid precision loss
        composited_float = (frame_restored.astype(np.float32) * mask_3channel +
                            original_frame.astype(np.float32) * (1 - mask_3channel))

        # Convert back to uint8
        composited = np.clip(composited_float, 0, 255).astype(np.uint8)

        return composited

    def resize_image_minimax(self, image, mask=False):
        """
        Resize image based on minimax internal resolution setting.
        - Downscales if the smaller side exceeds max_size.
        - Always ensures dimensions are multiples of 16 (rounded down).
        - Skips resizing if the output size would be identical.
        - Uses INTER_NEAREST for masks, INTER_AREA otherwise.
        """
        settings_mgr = get_settings_manager()
        max_size = settings_mgr.get_session_setting("minimax_resolution", 480)
        return core.resize_to_limit(image, max_size, multiple=16, mask=mask)

    def load_propainterx_model(self, parent_window=None):
        from propainterx.propainterx_pipeline import ProPainterXPipeline

        device = core.DeviceManager.get_device()
        core.DeviceManager.clear_cache()

        model_dir = "./checkpoints/propainterx/"
        propainter_ckpt = os.path.join(model_dir, "ProPainter.pth")
        flowcomp_ckpt = os.path.join(model_dir, "recurrent_flow_completion.pth")
        memfof_model_dir = os.path.join(model_dir, "memfof")

        # download models if they don't exist
        if not ensure_models(["propainter", "flow_completion", "memfof"], parent=parent_window):
            return False

        self.propainterx_pipeline = ProPainterXPipeline(
            device=device,
            propainter_ckpt=propainter_ckpt,
            flowcomp_ckpt=flowcomp_ckpt,
            memfof_model_dir=memfof_model_dir,
            fp16=True,
            clear_cache=core.DeviceManager.clear_cache,
        )
        self.propainterx_pipeline.load()
        return self.propainterx_pipeline

    def unload_propainterx_model(self):
        if self.propainterx_pipeline is not None:
            try:
                self.propainterx_pipeline.unload()
            except Exception as e:
                print(f"Warning: Could not unload ProPainterX model cleanly: {e}")
            self.propainterx_pipeline = None

        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded ProPainterX model")

    def resize_image_propainterx(self, image, mask=False):
        """
        Resize image based on the propainterx internal resolution setting.
        - Downscales if the smaller side exceeds max_size.
        - Always ensures dimensions are multiples of 8 (rounded down), as required by ProPainterX.
        - Skips resizing if the output size would be identical.
        - Uses INTER_NEAREST for masks, INTER_AREA otherwise.
        """
        settings_mgr = get_settings_manager()
        max_size = settings_mgr.get_session_setting("propainterx_resolution", 720)
        return core.resize_to_limit(image, max_size, multiple=8, mask=mask)

    def _load_all_frames_and_masks_propainterx(self, points_list, inpaint_grow=0, start_frame=0, end_frame=None, on_progress=None):
        """
        Load all frames and corresponding combined masks into memory, resized for ProPainterX.
        Unlike the MiniMax loader, frames/masks are left as raw uint8 (ProPainterX's own
        pipeline handles normalization and mask dilation internally).

        on_progress: optional callable(done, total) called periodically as frames load.

        Returns:
            tuple: (frames, masks)
                - frames: list of np.ndarray (uint8, HxWx3, RGB)
                - masks: list of np.ndarray (uint8, HxW, {0, 255})
        """
        frame_count = core.VideoInfo.total_frames
        if end_frame is None:
            end_frame = frame_count - 1
        extension = core.get_frame_extension()

        object_ids = sorted(list(set(p['object_id'] for p in points_list if 'object_id' in p)))
        if not object_ids:
            print("No objects found — returning empty frame/mask arrays.")
            return [], []

        frames = []
        masks = []
        total = end_frame - start_frame + 1

        for i, frame_number in enumerate(range(start_frame, end_frame + 1)):
            if on_progress is not None:
                on_progress(i, total)

            frame_path = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")

            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Combine masks for all objects on this frame (falls back to an
            # all-black mask, matching the frame's shape, if none are found)
            combined_mask, _ = core.load_combined_mask(frame_number, object_ids)
            if combined_mask is None:
                combined_mask = np.zeros(frame.shape[:2], np.uint8)

            combined_mask = core.apply_mask_postprocessing(combined_mask)

            if inpaint_grow != 0:
                combined_mask = core.grow_shrink(combined_mask, inpaint_grow)

            frame = self.resize_image_propainterx(frame)
            combined_mask = self.resize_image_propainterx(combined_mask, mask=True)
            combined_mask = ((combined_mask > 127) * 255).astype(np.uint8)

            frames.append(frame)
            masks.append(combined_mask)

        print(f"Loaded {len(frames)} frames and masks into memory.")
        return frames, masks

    def run_object_removal_propainterx(self, points, parent_window=None):
        """
        Run ProPainterX (inpainting) on all frames.
        Combines masks for all objects and loads all frames and masks into memory upfront.

        Args:
            points (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        from propainterx.propainterx_pipeline import CancelledError

        frame_count = core.VideoInfo.total_frames
        settings_mgr = get_settings_manager()
        self.propagated = False

        # Determine frame range to process
        start_frame, end_frame, frames_to_process = core.get_frame_range()

        print(f"Processing removal from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get settings
        grow = settings_mgr.get_session_setting("grow", 0)  # segmentation grow
        inpaint_grow = settings_mgr.get_session_setting("inpaint_grow", 0)  # object removal grow
        inpaint_grow = inpaint_grow + grow

        device = core.DeviceManager.get_device()
        extension = core.get_frame_extension()

        progress_dialog = QProgressDialog("Loading ProPainterX model...", "Cancel", 0, 0, parent_window)
        progress_dialog.setWindowTitle("Object Removal Progress")
        progress_dialog.setWindowModality(Qt.ApplicationModal)
        progress_dialog.show()
        QApplication.processEvents()

        if self.load_propainterx_model(parent_window=parent_window) is False:
            progress_dialog.close()
            return 0

        progress_dialog.setRange(0, 100)
        # Create output directory if it doesn't exist (don't clear existing frames)
        os.makedirs(core.removal_dir, exist_ok=True)

        print("Loading frames and masks...")

        def on_load_progress(done, total):
            percent = int((done / total) * 100) if total else 100
            progress_dialog.setLabelText(f"Loading frames and masks... ({done}/{total})")
            progress_dialog.setValue(percent)
            QApplication.processEvents()

        frames, masks = self._load_all_frames_and_masks_propainterx(
            points, inpaint_grow=inpaint_grow, start_frame=start_frame, end_frame=end_frame,
            on_progress=on_load_progress,
        )

        if not frames:
            print("No frames to process.")
            progress_dialog.close()
            return 0

        def on_progress(stage_idx, stage_count, label, done, total):
            percent = int((done / total) * 100) if total else 100
            progress_dialog.setLabelText(f"Stage {stage_idx + 1}/{stage_count}: {label} ({done}/{total})")
            progress_dialog.setValue(percent)
            QApplication.processEvents()

        def should_cancel():
            QApplication.processEvents()
            return progress_dialog.wasCanceled()

        def on_frame_done(idx, frame):
            frame_number = start_frame + idx
            composited = self.composite_removal_over_original(frame, frame_number, points)
            output_path = os.path.join(core.removal_dir, f"{frame_number:05d}.{extension}")
            frame_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, frame_bgr)

        try:
            self.propainterx_pipeline.run(
                frames, masks, on_progress=on_progress, on_frame_done=on_frame_done, should_cancel=should_cancel
            )
        except CancelledError:
            print("User cancelled ProPainterX processing.")
            self.propagated = False
            progress_dialog.close()
            self.unload_propainterx_model()
            return 0
        except RuntimeError as e:
            self.propagated = False
            progress_dialog.close()
            self.unload_propainterx_model()
            raise
        finally:
            del frames, masks

        if frame_count == frames_to_process:  # only set propagated if the whole video was processed
            self.propagated = True
        else:
            self.propagated = False
        print("Processing complete!")
        progress_dialog.close()
        self._notify('removal_complete')
        return 1

    def clear_removal(self):
        """Clear removal data"""
        if os.path.exists(core.removal_dir):
            shutil.rmtree(core.removal_dir)
        os.makedirs(core.removal_dir)
        self.propagated = False
        print("Object removal data cleared")
