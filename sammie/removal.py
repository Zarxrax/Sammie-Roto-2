# sammie/removal.py
import cv2
import os
import math
import numpy as np
import torch
import shutil
from tqdm import tqdm
from PySide6.QtWidgets import QProgressDialog, QApplication
from PySide6.QtCore import Qt
from sammie import core
from sammie.settings_manager import get_settings_manager
from sammie.model_downloader import ensure_models


class RemovalManager:
    """Manager for object removal operations"""

    def __init__(self):
        self.pipe = None
        self.propagated = False  # whether removal has been completed
        self.callbacks = []

    def add_callback(self, callback):
        """Add callback for removal events"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")

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
        """Unload the MiniMax model and clear cache"""
        if self.pipe is not None:
            # Explicitly delete the internal components
            del self.pipe.transformer
            del self.pipe.vae
            del self.pipe.scheduler
            self.pipe = None
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

        # Get in/out points from settings
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)

        # Determine frame range to process
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1

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
        try:
            with torch.no_grad():
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
            self.unload_minimax_model()

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

            # Combine masks for all objects on this frame
            combined_mask = np.zeros(frame.shape[:2], np.uint8)
            for object_id in object_ids:
                mask_path = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{object_id}.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)

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

        h, w = image.shape[:2]
        min_side = min(h, w)

        if min_side > max_size:
            # Downscale proportionally and align to multiple of 16 (rounded down)
            scale = max_size / min_side
            new_h = math.floor((h * scale) / 16) * 16
            new_w = math.floor((w * scale) / 16) * 16
        else:
            # Keep same size, just align down to multiple of 16
            new_h = math.floor(h / 16) * 16
            new_w = math.floor(w / 16) * 16

        # Only resize if necessary
        if (new_w, new_h) != (w, h):
            interpolation = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        return image

    def run_object_removal_cv(self, points_list, parent_window):
        """
        Run OpenCV object removal (inpainting) on all frames with points.
        Processes per frame instead of per object and combines masks for all objects.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        frame_count = core.VideoInfo.total_frames
        settings_mgr = get_settings_manager()

        # Get in/out points from settings
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)

        # Determine frame range to process
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1

        print(f"Processing removal from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        # Get settings
        inpaint_method = settings_mgr.get_session_setting("inpaint_method", "Telea")
        inpaint_radius = settings_mgr.get_session_setting("inpaint_radius", 3)
        grow = settings_mgr.get_session_setting("grow", 0)  # segmentation grow
        inpaint_grow = settings_mgr.get_session_setting("inpaint_grow", 0)  # object removal grow
        inpaint_grow = inpaint_grow + grow
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        # Convert method string to OpenCV constant
        if inpaint_method == "Telea":
            cv2_method = cv2.INPAINT_TELEA
        elif inpaint_method == "Navier-Stokes":
            cv2_method = cv2.INPAINT_NS
        else:
            print(f"Unknown inpaint method: {inpaint_method}, defaulting to Telea")
            cv2_method = cv2.INPAINT_TELEA

        # Get unique object IDs
        object_ids = sorted(list(set(p['object_id'] for p in points_list if 'object_id' in p)))
        if not object_ids:
            print("No objects found for removal")
            return 0

        # Create progress dialog
        progress_dialog = QProgressDialog("Running object removal...", "Cancel", 0, frames_to_process, parent_window)
        progress_dialog.setWindowTitle("Object Removal Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()

        # Terminal progress bar
        tqdm_bar = tqdm(total=frame_count, desc="Object Removal", unit="frame", ncols=80)

        # Create output directory if it doesn't exist (don't clear existing frames)
        os.makedirs(core.removal_dir, exist_ok=True)

        extension = core.get_frame_extension()
        operations_completed = 0

        # Process each frame in the range
        for frame_number in range(start_frame, end_frame + 1):
            if progress_dialog.wasCanceled():
                break

            frame_filename = os.path.join(core.frames_dir, f"{frame_number:05d}.{extension}")
            if not os.path.exists(frame_filename):
                operations_completed += 1
                tqdm_bar.update(1)
                progress_dialog.setValue(operations_completed)
                QApplication.processEvents()
                continue

            frame = cv2.imread(frame_filename)
            if frame is None:
                operations_completed += 1
                tqdm_bar.update(1)
                progress_dialog.setValue(operations_completed)
                QApplication.processEvents()
                continue

            # Combine masks from all objects on this frame
            combined_mask = np.zeros(frame.shape[:2], np.uint8)
            for object_id in object_ids:
                mask_filename = os.path.join(core.mask_dir, f"{frame_number:05d}", f"{object_id}.png")
                if os.path.exists(mask_filename):
                    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Skip if no mask present, copy original frame
            if not np.any(combined_mask):
                output_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, frame)
                operations_completed += 1
                tqdm_bar.update(1)
                progress_dialog.setValue(operations_completed)
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception:
                        pass
                    QApplication.processEvents()
                continue

            # Apply mask grow/shrink if requested
            if inpaint_grow != 0:
                combined_mask = core.grow_shrink(combined_mask, inpaint_grow)

            # Run inpainting
            try:
                result = cv2.inpaint(frame, combined_mask, inpaint_radius, cv2_method)
                output_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, result)

            except Exception as e:
                print(f"Error inpainting frame {frame_number}: {e}")
                output_filename = os.path.join(core.removal_dir, f"{frame_number:05d}.png")
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, frame)

            operations_completed += 1
            tqdm_bar.update(1)
            progress_dialog.setValue(operations_completed)

            # UI updates
            if frame_number % display_update_frequency == 0:
                progress_dialog.setValue(operations_completed)
                try:
                    parent_window.frame_slider.setValue(frame_number)
                except Exception as e:
                    print(f"Error updating display: {e}")
                QApplication.processEvents()

        # Finalize
        tqdm_bar.close()
        if progress_dialog.wasCanceled():
            print("Object removal cancelled — partial results kept.")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(frames_to_process)
            if frame_count == frames_to_process:  # only set propagated if entire video was processed
                self.propagated = True
            else:
                self.propagated = False
            print("Object removal completed successfully.")
            self._notify('removal_complete')
            return 1

    def clear_removal(self):
        """Clear removal data"""
        if os.path.exists(core.removal_dir):
            shutil.rmtree(core.removal_dir)
        os.makedirs(core.removal_dir)
        self.propagated = False
        print("Object removal data cleared")
