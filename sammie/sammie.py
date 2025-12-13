# sammie/sammie.py
import cv2
import os
import numpy as np
import shutil
import torch
import re
import glob
import zipfile
import math
import threading
import queue
import multiprocessing
from tqdm import tqdm
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication, QMessageBox
from sam2.build_sam import build_sam2_video_predictor
from matanyone.inference.inference_core import InferenceCore
from matanyone.utils.get_default_model import get_matanyone_model
from sammie.smooth import run_smoothing_model, prepare_smoothing_model
from sammie.duplicate_frame_handler import replace_similar_matte_frames
from sammie.settings_manager import get_settings_manager
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from minimax_remover.pipeline_minimax_remover import Minimax_Remover_Pipeline
from minimax_remover.transformer_minimax_remover import Transformer3DModel

# .........................................................................................
# Global variables
# .........................................................................................

temp_dir = "temp"
frames_dir = os.path.join(temp_dir, "frames")
mask_dir = os.path.join(temp_dir, "masks")
backup_dir = os.path.join(temp_dir, "masks_backup")
matting_dir = os.path.join(temp_dir, "matting")
removal_dir = os.path.join(temp_dir, "removal")
smoothing_model = None #global variable needed to avoid complexity of passing the model around

PALETTE = [
    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
    (128, 128, 128), (64, 0, 0), (191, 0, 0), (64, 128, 0), (191, 128, 0), (64, 0, 128),
    (191, 0, 128), (64, 128, 128), (191, 128, 128), (0, 64, 0), (128, 64, 0), (0, 191, 0),
    (128, 191, 0), (0, 64, 128), (128, 64, 128)
]

class VideoInfo:
    width = 0
    height = 0
    fps = 0
    total_frames = 0

class DeviceManager:
    _device = None

    @classmethod
    def setup_device(cls):
        """Detect and set up the best available device"""
        if cls._device is not None:
            return cls._device  # already set

        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        print("PyTorch version:", torch.__version__)

        settings_mgr = get_settings_manager()
        force_cpu = settings_mgr.get_app_setting("force_cpu", 0)

        if torch.cuda.is_available():
            cls._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            cls._device = torch.device("mps")
        else:
            cls._device = torch.device("cpu")
        
        if force_cpu:
            cls._device = torch.device("cpu")

        print(f"Using device: {cls._device}")

        if cls._device.type == "cuda":
            print("CUDA Compute Capability: ", torch.cuda.get_device_capability())
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        return cls._device

    @classmethod
    def get_device(cls):
        """Return the already initialized device (or setup if needed)"""
        if cls._device is None:
            return cls.setup_device()
        return cls._device

    @classmethod
    def clear_cache(cls):
        if cls._device is None:
            return
        if cls._device.type == "cuda":
            torch.cuda.empty_cache()
        elif cls._device.type == "mps":
            torch.mps.empty_cache()

class SamManager:
    def __init__(self):
        self.model = None
        self.predictor = None
        self.inference_state = None
        self.propagated = False # whether we have propagated the masks
        self.deduplicated = False # whether we have deduplicated the masks
        self.callbacks = []  # Add callbacks for segmentation events
        
    def add_callback(self, callback):
        """Add callback for segmentation events"""
        self.callbacks.append(callback)
    
    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except Exception as e:
                print(f"Callback error: {e}")
                
    # Load the sam2 model
    def load_segmentation_model(self):
        settings_mgr = get_settings_manager()
        sam_model = settings_mgr.get_app_setting("sam_model", "Base")
        DeviceManager.clear_cache()
        device = DeviceManager.get_device()
        if sam_model == "Large":
            print("Loaded SAM2 Large model")
            checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "../configs/sam2.1_hiera_l.yaml"
        elif sam_model == "Base":
            print("Loaded SAM2 Base model")
            checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "../configs/sam2.1_hiera_b+.yaml"
        self.predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)

    def offload_model_to_cpu(self):
        """Offload SAM2 model to CPU to free VRAM"""
        device = DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do
        
        if self.predictor is not None:
            self.predictor.to('cpu')
            DeviceManager.clear_cache()
            #print(f"SAM2 model offloaded to CPU")
    
    def load_model_to_device(self):
        """Load SAM2 model back to the active device"""
        device = DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do
        
        if self.predictor is not None:
            self.predictor.to(device)
            #print(f"SAM2 model loaded to {device}")

    def initialize_predictor(self):
        # initialize the predictor
        self.inference_state = self.predictor.init_state(video_path=frames_dir, async_loading_frames=True, offload_video_to_cpu=True)
    
    # Function to run segmentation and save the mask
    def segment_image(self, frame_number, object_id, input_points, input_labels):
        extension = get_frame_extension()
        frame_filename = os.path.join(frames_dir, f"{frame_number:05d}.{extension}")
        if os.path.exists(frame_filename):

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box( # returns a list of masks which includes all objects
                inference_state=self.inference_state,
                frame_idx=frame_number,
                obj_id=object_id,
                points=input_points,
                labels=input_labels
            )
            # Save the segmentation masks
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_filename = os.path.join(mask_dir, f"{frame_number:05d}", f"{out_obj_id}.png")
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                mask = (mask * 255).astype(np.uint8)
                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                cv2.imwrite(mask_filename, mask)
                
            # Notify that segmentation is complete
            self._notify('segmentation_complete', frame=frame_number, object_id=object_id, out_obj_ids=out_obj_ids)

    def replay_points(self, points_list):
        """Replay all points incrementally to rebuild masks"""
        frame_count = VideoInfo.total_frames
        
        
        for frame_number in range(frame_count):
            # Filter points for the current frame
            frame_points = [point for point in points_list if point['frame'] == frame_number]
            if not frame_points:
                continue

            # Group points by object_id
            frame_object_ids = {point['object_id'] for point in frame_points}
            for object_id in frame_object_ids:
                # Filter points for the current object ID
                filtered_points = [(point['x'], point['y'], point['positive']) 
                                   for point in frame_points if point['object_id'] == object_id]

                # Process points incrementally
                for i in range(1, len(filtered_points) + 1):
                    # Use only the first `i` points
                    subset_points = filtered_points[:i]
                    input_points = np.array([(x, y) for x, y, _ in subset_points], dtype=np.float32)
                    input_labels = np.array([1 if positive else 0 for _, _, positive in subset_points], dtype=np.int32)

                    try:
                        # Process the points with the segmentation model
                        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=frame_number,
                            obj_id=object_id,
                            points=input_points,
                            labels=input_labels
                        )

                    except Exception as e:
                        print(f"Error during prediction for frame {frame_number}, object {object_id}, points {i}: {e}")
                        continue

                    # Save masks (only on the final iteration for this object)
                    if i == len(filtered_points):
                        for j, out_obj_id in enumerate(out_obj_ids):
                            mask_filename = os.path.join(mask_dir, f"{frame_number:05d}", f"{out_obj_id}.png")
                            mask = (out_mask_logits[j] > 0.0).cpu().numpy().squeeze()
                            mask = (mask * 255).astype(np.uint8)
                            try:
                                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                                cv2.imwrite(mask_filename, mask)
                            except Exception as e:
                                print(f"Error saving mask for frame {frame_number}, object {out_obj_id}: {e}")
        
        # Notify that replay is complete
        self._notify('replay_complete')


    def track_objects(self, parent_window):
        frame_count = VideoInfo.total_frames
        progress_dialog = QProgressDialog("Tracking...", "Cancel", 0, 100, parent_window)
        progress_dialog.setWindowTitle("Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()
        
        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        # Figure out the frame range to track
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        if in_point is None:
            in_point = 0
        frames_to_track = None
        total_frames = frame_count
        if out_point is not None:
            frames_to_track = out_point - in_point
        if frames_to_track is not None:
            total_frames = frames_to_track + 1
    
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=in_point, max_frame_num_to_track=frames_to_track):
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_filename = os.path.join(mask_dir, f"{out_frame_idx:05d}", f"{out_obj_id}.png")
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                mask = (mask * 255).astype(np.uint8) # convert to uint8 before saving to file
                os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
                cv2.imwrite(mask_filename, mask)
            
        
            # Update progress dialog
            progress_dialog.setValue((out_frame_idx+1)*100/total_frames)
            
            # Update display at the specified frequency
            if out_frame_idx % display_update_frequency == 0:
                try:
                    parent_window.frame_slider.setValue(out_frame_idx)
                except Exception as e:
                    print(f"Error updating display: {e}")

            QApplication.processEvents()                
            # Check if user clicked cancel (if cancel button is enabled)
            if progress_dialog.wasCanceled():
                break
        
        if not progress_dialog.wasCanceled():
            progress_dialog.setValue(100)
            if total_frames == frame_count: # only set propagated if we're tracking the whole video
                self.propagated = True
            else:
                self.propagated = False
            print("Tracking completed")
            return 1
        else:
            progress_dialog.close()
            self.propagated = False
            print("Tracking cancelled")
            return 0

    def clear_tracking(self):
        """Clear tracking data by deleting all masks, this needs to be followed up by replay_points"""
        if os.path.exists(mask_dir):
            shutil.rmtree(mask_dir)
        os.makedirs(mask_dir)
        self.predictor.reset_state(self.inference_state)
        DeviceManager.clear_cache()
        if self.propagated:
            print("Tracking data cleared")
        self.propagated = False
        self.deduplicated = False

class MatAnyManager:
    
    def __init__(self):
        self.processor = None
        self.propagated = False # whether we have propagated the masks
        self.callbacks = []  # Add callbacks for matting events
        
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
                
    def load_matting_model(self, load_to_cpu=False):
        """Load the MatAnyone model and return processor"""
        DeviceManager.clear_cache()
        settings_mgr = get_settings_manager()
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        # Choose device based on parameter
        if load_to_cpu:
            device = torch.device('cpu')
        else:
            device = DeviceManager.get_device()
        checkpoint = "./checkpoints/matanyone.pth"
        matanyone = get_matanyone_model(checkpoint, device=device)
        print(f"Loaded MatAnyone model to {device} with max size {max_size}")
        
        # Initialize inference processor
        self.processor = InferenceCore(matanyone, cfg=matanyone.cfg, device=device)
        return self.processor
    
    def unload_matting_model(self):
        """Unload the MatAnyone model and clear cache"""
        self.processor = None
        DeviceManager.clear_cache()
        print("Unloaded MatAnyone model")
        
    def offload_model_to_cpu(self):
        """Offload MatAnyone model to CPU to free VRAM"""
        device = DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do
        
        if self.processor is not None and hasattr(self.processor, 'network'):
            self.processor.clear_memory()
            self.processor.network.to('cpu')
            DeviceManager.clear_cache()
            #print(f"MatAnyone model offloaded to CPU")
    
    def load_model_to_device(self):
        """Load MatAnyone model back to the active device"""
        device = DeviceManager.get_device()
        if device.type == 'cpu':
            return  # Already on CPU, nothing to do
        
        if self.processor is not None and hasattr(self.processor, 'network'):
            self.processor.clear_memory()
            self.processor.network.to(device)
            #print(f"MatAnyone model loaded to {device}")

    def _resize_image(self, image):
        """Resize image based on matting quality setting"""
        settings_mgr = get_settings_manager()
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        if max_size > 0:
            h, w = image.shape[:2]
            min_side = min(h, w)
            if min_side > max_size:
                scale = max_size / min_side
                new_h = int(h * scale)
                new_w = int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def _restore_image_size(self, image, original_size):
        """Restore image to original size"""
        original_h, original_w = original_size
        restored_image = cv2.resize(image, (original_h, original_w), interpolation=cv2.INTER_LINEAR)
        return restored_image
    
    @torch.inference_mode() 
    def run_matting(self, points_list, parent_window):
        """
        Run matting on all frames, using multiple keyframes for each object.
        
        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog
            
        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        if self.processor is None:
            print("Matting model not loaded")
            return 0
            
        DeviceManager.clear_cache()
        device = DeviceManager.get_device()
        frame_count = VideoInfo.total_frames
        
        # Get in/out points from settings
        settings_mgr = get_settings_manager()
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        
        # Determine frame range to process
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1
        
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")
        
        # Get unique object IDs from points list
        object_ids = sorted(list(set(point['object_id'] for point in points_list if 'object_id' in point)))
        if not object_ids:
            print("No objects found for matting")
            return 0
            
        # Find all keyframes for each object (within processing range)
        object_keyframes = {}
        for object_id in object_ids:
            keyframes = sorted(list(set(
                point['frame'] for point in points_list 
                if point.get('object_id') == object_id and start_frame <= point['frame'] <= end_frame
            )))
            if keyframes:
                object_keyframes[object_id] = keyframes
            else:
                print(f"No frames found for object {object_id} in range {start_frame}-{end_frame}")
                
        if not object_keyframes:
            print("No valid keyframes found for any objects in the specified range")
            return 0
            
        # Calculate total operations for progress tracking
        total_operations = 0
        for object_id, keyframes in object_keyframes.items():
            first_keyframe = keyframes[0]
            last_keyframe = keyframes[-1]
            
            # Operations before first keyframe (backward propagation to start_frame)
            total_operations += first_keyframe - start_frame
            
            # Operations between keyframes and after last keyframe to end_frame
            for i in range(len(keyframes)):
                if i == len(keyframes) - 1:
                    # Last keyframe to end_frame
                    total_operations += end_frame - keyframes[i] + 1
                else:
                    # Between consecutive keyframes
                    total_operations += keyframes[i + 1] - keyframes[i]
        
        # Create progress dialog
        progress_dialog = QProgressDialog("Running matting...", "Cancel", 0, 100, parent_window)
        progress_dialog.setWindowTitle("Matting Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()

        # Create tqdm progress bar
        pbar = tqdm(total=total_operations, desc="Matting Progress", unit="frame")

        # Create matting directory if it doesn't exist (don't clear existing data)
        os.makedirs(matting_dir, exist_ok=True)

        # Get list of image paths
        extension = get_frame_extension()
        images = []
        for frame_number in range(start_frame, end_frame + 1):
            image_filename = os.path.join(frames_dir, f"{frame_number:05d}.{extension}")
            if os.path.exists(image_filename):
                images.append(image_filename)

        operations_completed = 0

        # Process each object with its keyframes
        for object_id, keyframes in object_keyframes.items():
            if progress_dialog.wasCanceled():
                break
                
            #print(f"Processing object {object_id} with keyframes at: {keyframes}")
            pbar.set_description(f"Object {object_id}")
            
            # Process segments for this object
            success = self._process_object_with_keyframes(
                images, object_id, keyframes, end_frame + 1, device,
                progress_dialog, operations_completed, total_operations, pbar, parent_window,
                start_frame=start_frame
            )
            
            if not success:
                break
                
            # Update operations completed for this object
            first_keyframe = keyframes[0]
            operations_completed += first_keyframe - start_frame  # backward from first keyframe
            
            for i in range(len(keyframes)):
                if i == len(keyframes) - 1:
                    operations_completed += end_frame - keyframes[i] + 1  # last keyframe to end
                else:
                    operations_completed += keyframes[i + 1] - keyframes[i]  # between keyframes

        # Close tqdm progress bar
        pbar.close()

        # Final cleanup
        DeviceManager.clear_cache()
        
        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(100)
            if frame_count == frames_to_process:
                self.propagated = True # only set propagated to True if the entire video was processed
            else:
                self.propagated = False
            print("Matting completed")
            self._notify('matting_complete')
            return 1

    def _process_object_with_keyframes(self, images, object_id, keyframes, frame_count, device, progress_dialog, 
                                       operations_completed, total_operations, pbar, parent_window, start_frame=0):
        """
        Process a single object using multiple keyframes.
        
        Args:
            images: List of image paths
            object_id: ID of the object to process
            keyframes: Sorted list of keyframe indices for this object
            frame_count: Total number of frames
            device: Processing device
            progress_dialog: Progress dialog for user feedback
            operations_completed: Number of operations completed so far
            total_operations: Total operations for all objects
            pbar: tqdm progress bar
            parent_window: Parent window
            start_frame: Starting frame for the processing range
            
        Returns:
            bool: True if successful, False if cancelled or failed
        """
        first_keyframe = keyframes[0]
        
        # Load and validate the first keyframe mask
        mask, original_size = self._load_mask_for_matting(object_id, first_keyframe, device)
        if mask is None:
            return False
        
        # Special case for single frame
        if len(images) == 1:
            return self._process_single_frame(images[0], mask, object_id, original_size, device)
        
        current_operations = operations_completed
        
        # 1. Process backward from first keyframe to start_frame
        if first_keyframe > start_frame:
            success = self._process_backward(images, mask, object_id, first_keyframe,
                                        original_size, device, progress_dialog, current_operations, 
                                        total_operations, parent_window, pbar, start_frame_offset=start_frame)
            if not success:
                return False
            current_operations += first_keyframe - start_frame
        
        # 2. Process forward segments between keyframes
        for i in range(len(keyframes)):
            if progress_dialog.wasCanceled():
                return False
                
            current_keyframe = keyframes[i]
            
            # Load mask for current keyframe (refresh for each segment)
            mask, original_size = self._load_mask_for_matting(object_id, current_keyframe, device)
            if mask is None:
                print(f"Failed to load mask for object {object_id} at keyframe {current_keyframe}")
                return False
            
            # Determine end frame for this segment
            if i == len(keyframes) - 1:
                # Last keyframe - process to end of range
                end_frame = frame_count
            else:
                # Process to next keyframe (exclusive)
                end_frame = keyframes[i + 1]
            
            # Process this forward segment
            if end_frame > current_keyframe:
                success = self._process_forward(images, mask, object_id, current_keyframe, original_size,
                                                 device, progress_dialog, current_operations, total_operations, 
                                                 parent_window, end_frame, pbar, start_frame_offset=start_frame)
                if not success:
                    return False
                current_operations += end_frame - current_keyframe

        return True

    def _process_single_frame(self, frame_path, mask, object_id, original_size, device):
        """Process a single frame for matting"""
        try:
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._resize_image(img)
            img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

            output_prob = self.processor.step(img, mask, objects=[1])
            for i in range(10):  # Warmup iterations
                output_prob = self.processor.step(img, first_frame_pred=True)
                DeviceManager.clear_cache()

            mat = self.processor.output_prob_to_mask(output_prob)
            mat = mat.detach().cpu().numpy()
            mat = (mat * 255).astype(np.uint8)
            mat = self._restore_image_size(mat, original_size)

            mat_filename = os.path.join(matting_dir, f"00000", f"{object_id}.png")
            os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
            cv2.imwrite(mat_filename, mat)
            return True
            
        except Exception as e:
            print(f"Error processing single frame: {e}")
            return False
    
    def _process_forward(self, images, mask, object_id, start_frame, original_size, device, progress_dialog, operations_completed, 
                         total_operations, parent_window, end_frame=None, pbar=None, start_frame_offset=0):
        """
        Process frames forward from start_frame.
        
        Args:
            images: List of image paths
            mask: Initial mask tensor
            object_id: Object ID
            start_frame: Starting frame (inclusive)
            original_size: Original image size
            device: Processing device
            progress_dialog: Progress dialog
            operations_completed: Operations completed before this segment
            total_operations: Total operations
            parent_window: Parent window
            end_frame: Ending frame (exclusive). If None, process to end of images.
            pbar: tqdm progress bar
            start_frame_offset: Offset for mapping array indices to absolute frame numbers
            
        Returns:
            bool: True if successful, False if cancelled or failed
        """
        if end_frame is None:
            end_frame = start_frame + len(images)
        
        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        
        try:
            for frame_number in range(start_frame, end_frame):
                if progress_dialog.wasCanceled():
                    return False
                # Map absolute frame number to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    print(f"Warning: Frame {frame_number} out of range for images array")
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)

                if frame_number == start_frame:
                    # First frame - initialize with mask
                    output_prob = self.processor.step(img, mask, objects=[1])
                    for i in range(10):  # Warmup iterations
                        output_prob = self.processor.step(img, first_frame_pred=True)
                        DeviceManager.clear_cache()
                else:
                    # Subsequent frames - propagate
                    output_prob = self.processor.step(img)

                # Convert to matte
                mat = self.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(matting_dir, f"{frame_number:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                DeviceManager.clear_cache()
                
                # Update display at the specified frequency
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                # Update progress
                if pbar is not None:
                    pbar.update(1)
                current_progress = int(((operations_completed + (frame_number - start_frame) + 1) * 100) / total_operations)
                progress_dialog.setValue(current_progress)
                QApplication.processEvents()

            return True
            
        except Exception as e:
            print(f"Error in forward processing: {e}")
            return False
    
    
    def _process_backward(self, images, mask, object_id, start_frame, original_size, device, progress_dialog, 
                          operations_completed, total_operations, parent_window, pbar=None, start_frame_offset=0):
        """Process frames backward from start_frame"""

        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)

        try:
            for frame_number in range(start_frame, start_frame_offset - 1, -1):
                if progress_dialog.wasCanceled():
                    return False
                
                # Map absolute frame number to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    print(f"Warning: Frame {frame_number} out of range for images array")
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._resize_image(img)
                img = torch.tensor(img / 255., dtype=torch.float32, device=device).permute(2, 0, 1)
                if frame_number == start_frame:
                    # First frame - initialize with mask
                    output_prob = self.processor.step(img, mask, objects=[1])
                    for i in range(10):  # Warmup iterations
                        output_prob = self.processor.step(img, first_frame_pred=True)
                        DeviceManager.clear_cache()
                else:
                    # Subsequent frames - propagate
                    output_prob = self.processor.step(img)

                # Convert to matte
                mat = self.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(matting_dir, f"{frame_number:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                DeviceManager.clear_cache()
                
                # Update display at the specified frequency
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                # Update progress
                if pbar is not None:
                    pbar.update(1)
                operations_completed += 1
                progress_dialog.setValue(operations_completed * 100 // total_operations)
                QApplication.processEvents()
                
            return True
            
        except Exception as e:
            print(f"Error in backward processing: {e}")
            return False
    
    def _load_mask_for_matting(self, object_id, frame_number, device):
        """
        Load and validate a mask for MatAnyone processing.
        
        Args:
            object_id: ID of the object
            frame_number: Frame number
            device: Processing device
            
        Returns:
            tuple: (mask_tensor, original_size) or (None, None) if failed
        """
        mask_filename = os.path.join(mask_dir, f"{frame_number:05d}", f"{object_id}.png")
        if not os.path.exists(mask_filename):
            print(f"Mask not found for object {object_id} at frame {frame_number}: {mask_filename}")
            return None, None
            
        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            print(f"Mask is blank or invalid for object {object_id} at frame {frame_number}")
            return None, None
            
        original_size = mask.shape[1::-1]
        mask = self._resize_image(mask)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)
        
        return mask, original_size

    def clear_matting(self):
        """Clear matting data"""
        if os.path.exists(matting_dir):
            shutil.rmtree(matting_dir)
        os.makedirs(matting_dir)
        self.propagated = False
        print("Matting data cleared")

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
    
    def load_minimax_model(self):
        settings_mgr = get_settings_manager()
        minimax_vae_tiling = settings_mgr.get_session_setting("minimax_vae_tiling", False)
        DeviceManager.clear_cache()
        device = DeviceManager.get_device()

        """Load the minimax remover models."""
        model_path="./checkpoints/"
        vae = AutoencoderKLWan.from_pretrained(
            f"{model_path}/vae", 
            torch_dtype = torch.float16,
            device = device
        )
        transformer = Transformer3DModel.from_pretrained(
            f"{model_path}/transformer", 
            torch_dtype = torch.float16,
            device = device
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
        self.pipe = None
        DeviceManager.clear_cache()
        print("Unloaded MiniMax-Remover model")
        

    def run_object_removal_minimax(self, points, parent_window=None):
        """
        Run MiniMax-Remover (inpainting) on all frames.
        Combines masks for all objects and loads all frames and masks into memory upfront.
        
        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog
            
        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """

        frame_count = VideoInfo.total_frames
        settings_mgr = get_settings_manager()
        device = DeviceManager.get_device()
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
        blank_image = np.zeros((VideoInfo.height, VideoInfo.width, 1), dtype=np.uint8)
        blank_image = self.resize_image_minimax(blank_image, mask=True)
        resized_h, resized_w = blank_image.shape[:2]

        # Create progress dialog
        progress_dialog = QProgressDialog("Loading MiniMax-Remover model...", "Cancel", 0, 0, parent_window)
        progress_dialog.setWindowTitle("Object Removal Progress")
        progress_dialog.show()
        print(f"Loading MiniMax-Remover model to {device} with resolution {resized_w}x{resized_h}...")
        QApplication.processEvents()

        # Load model
        self.load_minimax_model()
        if self.pipe is None:
            print("Error loading MiniMax-Remover model")
            progress_dialog.close()
            return 0

        # Link progress dialog to pipeline
        self.pipe.progress_dialog = progress_dialog

        # Create output directory if it doesn't exist (don't clear existing frames)
        os.makedirs(removal_dir, exist_ok=True)

        # Load and prepare data
        print("Loading frames and masks...")
        QApplication.processEvents()
        frames, masks = self._load_all_frames_and_masks(points, inpaint_grow=inpaint_grow, start_frame=start_frame, end_frame=end_frame)
        
        # Pad frames
        pad_frames = (4 - (frames_to_process % 4)) % 4 + 1
        if pad_frames > 0:
            for _ in range(pad_frames):
                frames.append(frames[-1].copy())
                masks.append(masks[-1].copy())
        
        # Convert to tensors
        device = DeviceManager.get_device()
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
        
        # Remove padding and convert
        if pad_frames > 0:
            output = output[:frames_to_process]
        output = np.uint8(output * 255)
        
        # Save frames
        print("Saving frames...")
        progress_dialog.setLabelText("Saving frames...")
        QApplication.processEvents()
        extension = get_frame_extension()

        # Save processed frames   
        for i, frame in enumerate(output):
            frame_number = start_frame + i
            composited = self.composite_removal_over_original(frame, frame_number, points)
            output_path = os.path.join(removal_dir, f"{frame_number:05d}.{extension}")
            frame_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, frame_bgr)
        
        if frame_count == frames_to_process: # only set propagated if the whole video was processed
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
        frame_count = VideoInfo.total_frames
        if end_frame is None:
            end_frame = frame_count - 1
        extension = get_frame_extension()

        # Get all unique object IDs
        object_ids = sorted(list(set(p['object_id'] for p in points_list if 'object_id' in p)))
        if not object_ids:
            print("No objects found — returning empty frame/mask arrays.")
            return [], []

        frames = []
        masks = []

        for frame_number in range(start_frame, end_frame + 1):
            frame_path = os.path.join(frames_dir, f"{frame_number:05d}.{extension}")

            # Load frame
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Combine masks for all objects on this frame
            combined_mask = np.zeros(frame.shape[:2], np.uint8)
            for object_id in object_ids:
                mask_path = os.path.join(mask_dir, f"{frame_number:05d}", f"{object_id}.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Apply segmentation postprocessing (holes, dots, border_fix, grow)
            combined_mask = apply_mask_postprocessing(combined_mask)
        
            # Only apply the inpaint_grow portion (grow was already applied in apply_mask_postprocessing)
            if inpaint_grow != 0:
                combined_mask = grow_shrink(combined_mask, inpaint_grow)

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
        original_size = (VideoInfo.width, VideoInfo.height)  # (width, height) for cv2.resize
        
        # Load original full-resolution image
        original_frame = load_base_frame(frame_number)
        if original_frame is None:
            print(f"Warning: Could not load original frame {frame_number}, using processed frame only")
            return cv2.resize(processed_frame, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Resize processed frame to original size
        frame_restored = cv2.resize(processed_frame, original_size, interpolation=cv2.INTER_LINEAR)
        
        # Load original masks
        original_mask = load_masks_for_frame(frame_number, points, return_combined=True)
        
        if original_mask is None:
            # No mask found, use original frame
            return original_frame
        
        # Apply segmentation postprocessing (holes, dots, border_fix, grow)
        original_mask = apply_mask_postprocessing(original_mask)
        
        # Apply additional inpaint_grow
        inpaint_grow = settings_mgr.get_session_setting("inpaint_grow", 0)
        inpaint_grow = inpaint_grow + 21 # make it much larger to account for feathering
        
        # Only apply the inpaint_grow portion (grow was already applied in apply_mask_postprocessing)
        if inpaint_grow != 0:
            original_mask = grow_shrink(original_mask, inpaint_grow)
        
        # Apply feathering to the mask for smoother transitions
        # Gaussian blur creates a soft edge that blends better
        feather_radius = 10  # Adjust this value for more/less feathering
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
        frame_count = VideoInfo.total_frames
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
        grow = settings_mgr.get_session_setting("grow", 0) # segmentation grow
        inpaint_grow = settings_mgr.get_session_setting("inpaint_grow", 0) # object removal grow
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
        os.makedirs(removal_dir, exist_ok=True)

        extension = get_frame_extension()
        operations_completed = 0

        # Process each frame in the range
        for frame_number in range(start_frame, end_frame + 1):
            if progress_dialog.wasCanceled():
                break

            frame_filename = os.path.join(frames_dir, f"{frame_number:05d}.{extension}")
            if not os.path.exists(frame_filename):
                # Update progress even if frame missing
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
                mask_filename = os.path.join(mask_dir, f"{frame_number:05d}", f"{object_id}.png")
                if os.path.exists(mask_filename):
                    mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)

            # Skip if no mask present, copy original frame
            if not np.any(combined_mask):
                output_filename = os.path.join(removal_dir, f"{frame_number:05d}.png")
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
                combined_mask = grow_shrink(combined_mask, inpaint_grow)

            # Run inpainting
            try:
                result = cv2.inpaint(frame, combined_mask, inpaint_radius, cv2_method)

                output_filename = os.path.join(removal_dir, f"{frame_number:05d}.png")
                os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                cv2.imwrite(output_filename, result)

            except Exception as e:
                print(f"Error inpainting frame {frame_number}: {e}")
                output_filename = os.path.join(removal_dir, f"{frame_number:05d}.png")
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
            if frame_count == frames_to_process: # only set propagated if entire video was processed
                self.propagated = True
            else:
                self.propagated = False
            print("Object removal completed successfully.")
            self._notify('removal_complete')
            return 1
        
    def clear_removal(self):
        """Clear removal data"""
        if os.path.exists(removal_dir):
            shutil.rmtree(removal_dir)
        os.makedirs(removal_dir)
        self.propagated = False
        print("Object removal data cleared")

class PointManager:
    def __init__(self):
        self.points = []  # List of dicts: {'frame': int, 'object_id': int, 'positive': bool, 'x': int, 'y': int}
        self.callbacks = []  # Callbacks for when points change
    
    def add_callback(self, callback):
        """Add callback for point changes"""
        self.callbacks.append(callback)
    
    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except Exception as e:
                print(f"Point callback error: {e}")
    
    def add_point(self, frame, object_id, positive, x, y):
        """Add a point"""
        point = {'frame': frame, 'object_id': object_id, 'positive': positive, 'x': x, 'y': y}
        self.points.append(point)
        self._notify('add', point=point)
        settings_mgr = get_settings_manager()
        settings_mgr.save_points(self.points)
        return point
    
    def remove_point(self, frame, object_id, x, y):
        """Remove a specific point"""
        before_count = len(self.points)
        point_to_remove = None
        
        # Find the matching point
        for i, point in enumerate(self.points):
            if (point['frame'] == frame and 
                point['object_id'] == object_id and 
                point['x'] == x and 
                point['y'] == y):
                point_to_remove = self.points.pop(i)
                break
        
        if point_to_remove:
            # Remove the corresponding mask file
            #mask_filename = os.path.join(mask_dir, f'{frame:05d}', f'{object_id}.png')
            #if os.path.exists(mask_filename):
            #    os.remove(mask_filename)
            
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('remove_point', point=point_to_remove)
            return point_to_remove
        return None
    
    def remove_last(self):
        """Remove last point"""
        if self.points:
            point = self.points.pop()
            mask_filename = os.path.join(mask_dir, f'{point["frame"]:05d}', f'{point["object_id"]}.png')
            if os.path.exists(mask_filename):
                os.remove(mask_filename)
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('remove_last', point=point)
            return point
        return None
    
    def clear_all(self):
        """Clear all points"""
        if self.points:  # Only notify if there were points to clear
            self.points.clear()
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('clear_all')
    
    def clear_frame(self, frame):
        """Clear points for a frame"""      
        before_count = len(self.points)
        points_to_remove = [p for p in self.points if p['frame'] == frame]
        self.points = [p for p in self.points if p['frame'] != frame]
        removed_count = before_count - len(self.points)
        
        if removed_count > 0:
            # Remove mask files for this frame
            frame_mask_dir = os.path.join(mask_dir, f"{frame:05d}")
            if os.path.exists(frame_mask_dir):
                shutil.rmtree(frame_mask_dir)
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('clear_frame', frame=frame, count=removed_count, points=points_to_remove)
        return removed_count
    
    def clear_object(self, object_id):
        """Clear points for an object"""
        before_count = len(self.points)
        points_to_remove = [p for p in self.points if p['object_id'] == object_id]
        self.points = [p for p in self.points if p['object_id'] != object_id]
        removed_count = before_count - len(self.points)
        
        if removed_count > 0:
            # Remove mask files for this object across all frames
            for point in points_to_remove:
                mask_filename = os.path.join(mask_dir, f'{point["frame"]:05d}', f'{object_id}.png')
                matting_filename = os.path.join(matting_dir, f'{point["frame"]:05d}', f'{object_id}.png')
                if os.path.exists(mask_filename):
                    os.remove(mask_filename)
                if os.path.exists(matting_filename):
                    os.remove(matting_filename)
            settings_mgr = get_settings_manager()
            settings_mgr.save_points(self.points)
            self._notify('clear_object', object_id=object_id, count=removed_count, points=points_to_remove)
        return removed_count

    def get_sam2_points(self, frame, object_id=None):
        """Get points in SAM2 format: (coordinates, labels)"""
        frame_points = [p for p in self.points if p['frame'] == frame]
        if object_id is not None:
            frame_points = [p for p in frame_points if p['object_id'] == object_id]
        
        if not frame_points:
            return [], []
        
        coordinates = [[p['x'], p['y']] for p in frame_points]
        labels = [1 if p['positive'] else 0 for p in frame_points]
        return coordinates, labels
    
    def get_points_for_frame(self, frame):
        """Get all points for a frame"""
        return [p for p in self.points if p['frame'] == frame]
    
    def get_all_points(self):
        """Get all points"""
        return self.points.copy()


def get_frame_extension():
    """Get the frame file extension from session settings, fallback to PNG"""
    settings_mgr = get_settings_manager()
    frame_format = settings_mgr.get_session_setting("frame_format", "png")
    return frame_format



def update_image(slider_value, view_options, points, return_numpy=False, object_id_filter=None):
    """Main image update function - delegates to specific view handlers
    
    Args:
        slider_value: Frame number
        view_options: Dictionary of view options
        points: List of point dictionaries
        return_numpy: If True, return numpy array; if False, return QPixmap
        object_id_filter: If specified, only process masks for this object ID
    
    Returns:
        QPixmap or numpy array depending on return_numpy parameter
    """
    view_mode = view_options.get("view_mode", "Segmentation-Edit") #Defaults to Segmentation-Edit

    # Dispatch to appropriate view handler
    if view_mode == "Segmentation-Edit":
        return _handle_segmentation_edit_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-Matte":
        return _handle_segmentation_matte_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-BGcolor":
        return _handle_segmentation_bgcolor_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Segmentation-Alpha":
        return _handle_segmentation_alpha_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-Matte":
        return _handle_matting_matte_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-BGcolor":
        return _handle_matting_bgcolor_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "Matting-Alpha":
        return _handle_matting_alpha_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "ObjectRemoval":
        return _handle_object_removal_view(slider_value, view_options, points, return_numpy, object_id_filter)
    elif view_mode == "None":
        return _handle_none_view(slider_value, return_numpy)
    else:
        print(f"Unknown view mode: {view_mode}")
        return None


def load_base_frame(frame_number):
    """Load the base frame image from disk"""
    extension = get_frame_extension()
    frame_filename = os.path.join(frames_dir, f"{frame_number:05d}.{extension}")
    if os.path.exists(frame_filename):
        image = cv2.imread(frame_filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: 
        print(f"{frame_filename} not found")
        return None
        
def load_removal_frame(frame_number):
    """
    Load the object removal frame image from disk
    If the frame does not exist, load the base frame instead
    """
    frame_filename = os.path.join(removal_dir, f"{frame_number:05d}.png")
    if os.path.exists(frame_filename):
        image = cv2.imread(frame_filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else: 
        return load_base_frame(frame_number)

def load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=None, folder=mask_dir):
    """
    Load masks for a frame, returning either individual masks or a combined mask.
    
    Args:
        frame_number (int): Frame number to load masks for
        points (list): List of point dictionaries containing object_id information
        return_combined (bool): If True, return single combined mask. If False, return dict of individual masks.
        object_id_filter (int): only load masks for a specific object id
        folder: which mask folder to get images from, mask_dir or matting_dir
    
    Returns:
        If return_combined=True: Single numpy array (grayscale) or None if no masks
        If return_combined=False: Dict {object_id: mask_array} or empty dict if no masks
    """
    # Get unique object IDs from points
    object_ids = list(set(p['object_id'] for p in points if 'object_id' in p))
    
    # Filter by specific object ID if requested
    if object_id_filter is not None:
        object_ids = [obj_id for obj_id in object_ids if obj_id == object_id_filter]
        
    if not object_ids:
        return None if return_combined else {}
    
    individual_masks = {}
    
    # Load each mask file
    for object_id in object_ids:
        mask_filename = os.path.join(folder, f"{frame_number:05d}", f"{object_id}.png")
        if os.path.exists(mask_filename):
            mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
            if mask is not None: #and np.any(mask):  # Ensure mask is not blank
                individual_masks[object_id] = mask
            #else:
            #    print(f"Warning: Mask file {mask_filename} is blank or corrupted")
        else:
            # if mask doesnt exist, create a blank frame
            individual_masks[object_id] = np.zeros((VideoInfo.height, VideoInfo.width), dtype=np.uint8)
    
    if not individual_masks:
        return None if return_combined else {}
    
    if return_combined:
        # Combine all masks into a single mask (union operation)
        combined_mask = np.zeros((VideoInfo.height, VideoInfo.width), dtype=np.uint8)
        for mask in individual_masks.values():
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        return combined_mask
    else:
        return individual_masks


def _convert_to_qpixmap(image):
    """Convert NumPy array to QPixmap"""
    if image is None:
        return QPixmap.fromImage(QImage())
    image = image.copy()
    height, width = image.shape[:2]

    if len(image.shape) == 2:  # Grayscale
        bytes_per_line = width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

    elif image.shape[2] == 3:  # RGB
        bytes_per_line = 3 * width
        q_image =  QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
    else: #RGBA
        bytes_per_line = 4 * width
        q_image =  QImage(image.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
    
    return QPixmap.fromImage(q_image)


def _handle_none_view(frame_number, return_numpy=False):
    """Handle None view"""
    image = load_base_frame(frame_number)
    if image is None:
        return None
    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)

def _handle_segmentation_edit_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Edit view"""
    image = load_base_frame(frame_number)
    if image is None:
        return None
    
    # Apply postprocessing to masks before display
    image = apply_postprocessing_to_display(image, frame_number, points, view_options, object_id_filter)
    
    # Get highlighted point from view options if present
    highlighted_points = view_options.get('highlighted_point', None)

    # Make a copy of the selected points list to avoid modifying the original 
    # Also handle case where it might be None or a single dict
    if highlighted_points is None:
        highlighted_points = None
    elif isinstance(highlighted_points, list):
        highlighted_points = highlighted_points.copy()
    else:
        # Single point dict - convert to list
        highlighted_points = [highlighted_points]

    # Always show points in edit view
    image = draw_points(image, frame_number, points, highlighted_points)
    
    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)

def _handle_segmentation_matte_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Matte view"""
    # Load combined mask
    mask = load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        return None
        
    # Apply postprocessing
    mask = apply_mask_postprocessing(mask)
    
    mask_3channel = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel
    
    if view_options.get("antialias", True):
        global smoothing_model
        if smoothing_model is None:
            load_smoothing_model()
        
        if smoothing_model is not None:
            device = DeviceManager.get_device()
            mask_3channel = run_smoothing_model(mask_3channel, smoothing_model, device)
    
    if return_numpy:
        return mask_3channel
    else:
        return _convert_to_qpixmap(mask_3channel)

def _handle_segmentation_bgcolor_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-BGcolor view"""
    image = load_base_frame(frame_number)
    if image is None:
        return None
        
    # Load combined mask
    mask = load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        if return_numpy:
            return image
        else:
            return _convert_to_qpixmap(image)
        
    # Apply postprocessing
    mask = apply_mask_postprocessing(mask)
        
    mask_3channel = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel
        
    if view_options.get("antialias", True):
        global smoothing_model
        if smoothing_model is None:
            load_smoothing_model()
        
        if smoothing_model is not None:
            device = DeviceManager.get_device()
            mask_3channel = run_smoothing_model(mask_3channel, smoothing_model, device)
    
    # Get bgcolor from settings and convert to numpy array
    bgcolor = view_options.get("bgcolor", (0, 255, 0))
    bgcolor_bgr = np.array(bgcolor)

    # Apply background color effect
    alpha_channel = mask_3channel / 255.0
    image = (image * alpha_channel) + (bgcolor_bgr * (1 - alpha_channel))
    image = image.astype(np.uint8)
    
    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)
    
def _handle_segmentation_alpha_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Segmentation-Alpha view - updated to use centralized loading"""
    image = load_base_frame(frame_number)
    if image is None:
        return None
    
    # Convert to RGBA
    image_rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    image_rgba[:, :, :3] = image  # RGB channels
    image_rgba[:, :, 3] = 255     # Alpha channel (fully opaque initially)
    
    # Load combined mask
    mask = load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        if return_numpy:
            return image_rgba
        else:
            return _convert_to_qpixmap(image_rgba)
        
    # Apply postprocessing
    mask = apply_mask_postprocessing(mask)

    if view_options.get("antialias", True):
        global smoothing_model
        if smoothing_model is None:
            load_smoothing_model()
        
        if smoothing_model is not None:
            device = DeviceManager.get_device()
            mask_3channel = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel
            mask_3channel = run_smoothing_model(mask_3channel, smoothing_model, device)
            mask = mask_3channel[:, :, 0] # Convert back to single channel - take first channel (all 3 should be identical)
    
    # Apply alpha channel
    image_rgba[:, :, 3] = mask
    
    if return_numpy:
        return image_rgba
    else:
        return _convert_to_qpixmap(image_rgba)

def _handle_matting_matte_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-Matte view"""
    # Load combined mask
    mask = load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter, folder=matting_dir)
    if mask is None:
        return None
        
    # Apply postprocessing
    mask = apply_matany_postprocessing(mask)
    
    mask_3channel = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel
    
    if return_numpy:
        return mask_3channel
    else:
        return _convert_to_qpixmap(mask_3channel)

def _handle_matting_bgcolor_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-BGcolor view"""
    image = load_base_frame(frame_number)
    if image is None:
        return None
        
    # Load combined mask
    mask = load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter, folder=matting_dir)
    if mask is None:
        if return_numpy:
            return image
        else:
            return _convert_to_qpixmap(image)
        
    # Apply postprocessing
    mask = apply_matany_postprocessing(mask)
        
    mask_3channel = np.stack([mask] * 3, axis=-1)  # Convert to 3-channel
    
    # Get bgcolor from settings and convert to numpy array
    bgcolor = view_options.get("bgcolor", (0, 255, 0))
    bgcolor_bgr = np.array(bgcolor)

    # Apply apply background color effect
    alpha_channel = mask_3channel / 255.0
    image = (image * alpha_channel) + (bgcolor_bgr * (1 - alpha_channel))
    image = image.astype(np.uint8)
    
    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)
    
def _handle_matting_alpha_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Matting-Alpha view"""
    image = load_base_frame(frame_number)
    if image is None:
        return None
    
    # Convert to RGBA
    image_rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    image_rgba[:, :, :3] = image  # RGB channels
    image_rgba[:, :, 3] = 255     # Alpha channel (fully opaque initially)
    
    # Load combined mask
    mask = load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter, folder=matting_dir)
    if mask is None:
        if return_numpy:
            return image_rgba
        else:
            return _convert_to_qpixmap(image_rgba)
        
    # Apply postprocessing
    mask = apply_matany_postprocessing(mask)
    
    # Apply alpha channel
    image_rgba[:, :, 3] = mask
    
    if return_numpy:
        return image_rgba
    else:
        return _convert_to_qpixmap(image_rgba)

def _handle_object_removal_view(frame_number, view_options, points, return_numpy=False, object_id_filter=None):
    """Handle Object Removal view"""
    image = load_removal_frame(frame_number)
    if image is None:
        return None
    
    # Load combined mask
    mask = load_masks_for_frame(frame_number, points, return_combined=True, object_id_filter=object_id_filter)
    if mask is None:
        return None
        
    # Apply postprocessing from segmentation tab first
    mask = apply_mask_postprocessing(mask)

    # Apply additional grow/shrink
    settings_mgr = get_settings_manager()
    grow = settings_mgr.get_session_setting("inpaint_grow", 0)
    mask = grow_shrink(mask, grow)
    
    # Draw mask overlay
    if view_options.get("show_removal_mask", True):
        image = draw_removal_overlay(image, mask)
    
    if return_numpy:
        return image
    else:
        return _convert_to_qpixmap(image)

def draw_masks(image, processed_masks):
    """Draw masks on the current frame (expects preprocessed masks)"""
    if not processed_masks:
        return image

    combined_colored_mask = np.zeros_like(image, dtype=np.uint8)
    mask_binary = np.zeros(image.shape[:2], dtype=bool)

    for object_id, mask in processed_masks.items():
        color = np.array(PALETTE[object_id % len(PALETTE)], dtype=np.uint8)
        mask_bin = mask > 0
        mask_binary |= mask_bin
        combined_colored_mask[mask_bin] = color

    if np.any(mask_binary):
        overlay = image.copy()
        overlay[mask_binary] = cv2.addWeighted(
            image[mask_binary], 0.5,
            combined_colored_mask[mask_binary], 0.5, 0
        )
        return overlay
    else:
        return image

def draw_removal_overlay(image, mask):
    """Draw masked overlay on the current frame for object removal"""

    # Normalize mask to 0–1 range
    mask_norm = mask.astype(np.float32) / 255.0
    mask_3ch = np.dstack([mask_norm] * 3)

    # Create a color layer
    color_layer = np.full_like(image, (255,255,255), dtype=np.uint8)

    # Blend color layer and original image where mask > 0
    overlay = np.where(
        (mask_3ch > 0),
        cv2.addWeighted(image, 1 - 0.5, color_layer, 0.5, 0),
        image
    )

    return overlay

def draw_contours(image, processed_masks):
    """Draw colored contours on the current frame (expects preprocessed masks)"""
    if not processed_masks:
        return image

    overlay = image.copy()
    kernel = np.ones((3, 3), np.uint8) # kernel size controls outline thickness

    for object_id, mask in processed_masks.items():
        edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        border_color = PALETTE[object_id % len(PALETTE)]
        overlay[edges > 0] = border_color

    return overlay


def draw_points(image, frame_number, points, highlighted_points=None):
    """Draw points on image, with optional highlighting"""
    # Filter once for current frame
    frame_points = [p for p in points if p['frame'] == frame_number]
    if not frame_points:
        return image
    
    is_highlighted = False

    # Get the object_id of the highlighted point (if any)
    # highlighted_object_id = highlighted_point.get('object_id') if highlighted_point else None
    
    for point in frame_points:
        is_highlighted = False
        if highlighted_points:
            for point_in_highlighted_list in highlighted_points:
                # Check if the current point is the same as the one in the highlighted list one
                is_highlighted = (
                    point_in_highlighted_list['frame'] == point['frame'] and
                    point_in_highlighted_list['x'] == point['x'] and
                    point_in_highlighted_list['y'] == point['y']
                )
                if is_highlighted:
                    highlighted_points.remove(point_in_highlighted_list)
                    break

        if is_highlighted:
            # Draw highlighted point with different colors/sizes
            center = (point['x'], point['y'])
            
            # Larger circle for highlight
            cv2.circle(image, center, 9, (0, 128, 255), 3)  # Thick blue outline
            #cv2.circle(image, center, 7, (0, 0, 0), 2)   # Black middle ring

            point_color = (0, 255, 0) if point['positive'] else (255, 0, 0)
            center = (point['x'], point['y'])

            # Draw yellow outline (radius 5, thickness 2)
            cv2.circle(image, center, 5, (255, 255, 0), 2)

            # Draw filled circle (radius 4, filled)
            cv2.circle(image, center, 4, point_color, -1)

        else:
            # Draw regular points
            point_color = (0, 255, 0) if point['positive'] else (255, 0, 0)
            center = (point['x'], point['y'])

            # Draw yellow outline (radius 5, thickness 2)
            cv2.circle(image, center, 5, (255, 255, 0), 2)

            # Draw filled circle (radius 4, filled)
            cv2.circle(image, center, 4, point_color, -1)

    return image
    

def apply_postprocessing_to_display(image, frame_number, points, view_options, object_id_filter=None):
    """Apply postprocessing to masks and draw them on the image for display"""
    # Load and preprocess masks once
    raw_masks = load_masks_for_frame(
        frame_number, points, return_combined=False, object_id_filter=object_id_filter
    )

    if raw_masks:
        processed_masks = {
            object_id: apply_mask_postprocessing(mask) for object_id, mask in raw_masks.items()
        }
    else:
        processed_masks = {}

    # Apply masks if enabled
    if view_options.get("show_masks", True):
        image = draw_masks(image, processed_masks)

    # Apply outlines if enabled
    if view_options.get("show_outlines", True):
        image = draw_contours(image, processed_masks)

    return image


def apply_mask_postprocessing(mask):
    """Apply postprocessing to a mask using current session settings"""
    settings_mgr = get_settings_manager()
    
    # Get current postprocessing parameters
    holes = settings_mgr.get_session_setting("holes", 0)
    dots = settings_mgr.get_session_setting("dots", 0)
    border_fix = settings_mgr.get_session_setting("border_fix", 0)
    grow = settings_mgr.get_session_setting("grow", 0)
    
    # Apply postprocessing in order
    if holes > 0:
        mask = fill_small_holes(mask, holes)
    
    if dots > 0:
        mask = remove_small_dots(mask, dots)
    
    if border_fix > 0:
        mask = apply_border_fix(mask, border_fix)
    
    if grow != 0:
        mask = grow_shrink(mask, grow)
    
    return mask

def apply_matany_postprocessing(mask):
    """Apply postprocessing to MatAnyone results using current session settings"""
    settings_mgr = get_settings_manager()
    
    # Get current postprocessing parameters
    grow = settings_mgr.get_session_setting("matany_grow", 0)
    gamma = settings_mgr.get_session_setting("matany_gamma", 1.0)
    
    # Apply postprocessing in order
    if grow != 0:
        mask = grow_shrink(mask, grow)
    
    if gamma != 1.0:
        mask = change_gamma(mask, gamma)
    
    return mask

# Mask postprocessing - holes
def fill_small_holes(mask, holes_value):
    max_hole_area = holes_value**2
    filled_mask = mask.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over contours and fill small holes
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area <= max_hole_area and hierarchy[0][i][3] != -1:  # Check if it's a hole (child contour)
            # Fill the hole by drawing the contour on the original mask
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filled_mask

def remove_small_dots(mask, dots_value):
    max_dot_area = dots_value**2
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned_mask = np.zeros_like(mask)
    for label in range(1, num_labels):  # skip background
        if stats[label, cv2.CC_STAT_AREA] > max_dot_area:
            cleaned_mask[labels == label] = 255

    return cleaned_mask


# Mask postprocessing - grow/shrink
def grow_shrink(mask, grow_value):
    kernel = np.ones((abs(grow_value)+1, abs(grow_value)+1), np.uint8)
    if grow_value > 0:
        return cv2.dilate(mask, kernel, iterations=1)
    elif grow_value < 0:
        return cv2.erode(mask, kernel, iterations=1)
    else:
        return mask

# Mask postprocessing - border fix
def apply_border_fix(mask, border_size):
    if border_size == 0:
        return mask
    else: 
        height, width = mask.shape
        y_start = border_size
        y_end = height - border_size
        x_start = border_size
        x_end = width - border_size
        return cv2.copyMakeBorder(mask[y_start:y_end, x_start:x_end], border_size, border_size, border_size, border_size, cv2.BORDER_REPLICATE, value=None)


def change_gamma(mask, gamma_value):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma_value
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(mask, table)


def deduplicate_masks(parent_window):
    """Deduplicate similar masks using settings threshold"""
    settings_mgr = get_settings_manager()
    threshold = settings_mgr.app_settings.dedupe_threshold
    return replace_similar_matte_frames(parent_window, threshold)

def remove_backup_mattes():
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    

def load_video(video_file, parent_window):
    """Load video and save frames as images using multi-threaded writers"""
    # Create temp directories to save the frames, delete if already exists
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(frames_dir)
    os.makedirs(mask_dir)
    os.makedirs(matting_dir)
    print(f"Loading video: {video_file}")

    progress_dialog = QProgressDialog("Loading video...", "Cancel", 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()

    cap = cv2.VideoCapture(video_file)
    VideoInfo.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VideoInfo.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    VideoInfo.fps = cap.get(cv2.CAP_PROP_FPS)
    VideoInfo.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = VideoInfo.total_frames

    frame_count = 0
    settings_mgr = get_settings_manager()
    frame_format = settings_mgr.get_app_setting("frame_format", "png")

    # --- Threaded frame writing setup ---
    save_q = queue.Queue(maxsize=100)  # limit queue size to avoid memory overflow
    num_workers = max(2, multiprocessing.cpu_count() // 2)  # use half of CPU cores

    def save_worker():
        while True:
            item = save_q.get()
            if item is None:
                save_q.task_done()
                break
            path, frame = item
            try:
                cv2.imwrite(path, frame)
            except Exception as e:
                print(f"Error writing {path}: {e}")
            save_q.task_done()

    # Start multiple background writer threads
    writers = []
    for _ in range(num_workers):
        t = threading.Thread(target=save_worker, daemon=True)
        t.start()
        writers.append(t)

    # --- Read frames and enqueue for saving ---
    with tqdm(total=total_frames) as progress:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_filename = os.path.join(frames_dir, f"{frame_count:05d}.{frame_format}")
            save_q.put((frame_filename, frame))  # enqueue instead of direct save
            frame_count += 1
            progress.update(1)
            progress_dialog.setValue(frame_count * 100 / total_frames)
            QApplication.processEvents()

            # Check for cancel
            if progress_dialog.wasCanceled():
                cap.release()
                # Empty the queue and stop the workers
                while not save_q.empty():
                    try:
                        save_q.get_nowait()
                    except queue.Empty:
                        pass
                # Signal threads to stop
                for _ in writers:
                    save_q.put(None)
                for t in writers:
                    t.join()
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                progress_dialog.close()
                print("Operation cancelled by user.")
                return 0

        cap.release()

    # Wait for all frames to finish writing
    save_q.join()

    # Stop all worker threads cleanly
    for _ in writers:
        save_q.put(None)
    for t in writers:
        t.join()

    progress_dialog.setValue(100)
    progress_dialog.close()

    # The actual number of frames processed can differ from what cv2 reported
    VideoInfo.total_frames = frame_count
    return frame_count


# Function to process the video and save frames as PNGs
def load_video_old(video_file, parent_window):

    # Create temp directories to save the frames, delete if already exists
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(frames_dir)
    os.makedirs(mask_dir)
    os.makedirs(matting_dir)
    print(f"Loading video: {video_file}")
    
    progress_dialog = QProgressDialog("Loading video...", "Cancel", 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()

    cap = cv2.VideoCapture(video_file)
    VideoInfo.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    VideoInfo.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    VideoInfo.fps = cap.get(cv2.CAP_PROP_FPS)
    VideoInfo.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = VideoInfo.total_frames

    frame_count = 0
    settings_mgr = get_settings_manager()
    frame_format = settings_mgr.get_app_setting("frame_format", "png")
    
    # Read and save frames
    with tqdm(total=total_frames) as progress:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                progress_dialog.setValue(100)
                break
            frame_filename = os.path.join(frames_dir, f"{frame_count:05d}.{frame_format}")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
            progress.update(1)
            progress_dialog.setValue(frame_count*100/total_frames)
            QApplication.processEvents()
            
            # Check if user clicked cancel (if cancel button is enabled)
            if progress_dialog.wasCanceled():
                cap.release()
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                progress_dialog.close()
                return 0
            
        cap.release()
        
    # the actual number of frames processed can differ from what cv2 reported
    VideoInfo.total_frames = frame_count
    return frame_count

def detect_image_sequence(image_path):
    """
    Detect if an image is part of a sequence based on common naming patterns.
    Returns (is_sequence, sequence_files) or (False, [])
    """
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    # Common sequence patterns to check
    patterns = [
        r'^(.+?)(\d{4,})$',  # name0001, name0002, etc. (4+ digits)
        r'^(.+?)(\d{3})$',   # name001, name002, etc. (3 digits)
        r'^(.+?)(\d{2})$',   # name01, name02, etc. (2 digits)
        r'^(.+?)_(\d+)$',    # name_1, name_2, etc. (underscore separator)
        r'^(.+?)\-(\d+)$',   # name-1, name-2, etc. (dash separator)
        r'^(.+?)\.(\d+)$',   # name.1, name.2, etc. (dot separator)
    ]
    
    for pattern in patterns:
        match = re.match(pattern, name)
        if match:
            base_name = match.group(1)
            
            # Create glob pattern to find all files with this naming scheme
            if pattern == r'^(.+?)(\d{4,})$':
                glob_pattern = f"{base_name}*{ext}"
            elif pattern == r'^(.+?)(\d{3})$':
                glob_pattern = f"{base_name}???{ext}"
            elif pattern == r'^(.+?)(\d{2})$':
                glob_pattern = f"{base_name}??{ext}"
            else:  # underscore, dash, or dot patterns
                separator = pattern.split('(\\d+)')[0][-1]  # Get the separator character
                glob_pattern = f"{base_name}{separator}*{ext}"
            
            # Find all matching files
            search_path = os.path.join(directory, glob_pattern)
            potential_files = glob.glob(search_path)
            
            # Filter files that actually match the pattern and sort them
            sequence_files = []
            for file_path in potential_files:
                file_name = os.path.basename(file_path)
                file_base = os.path.splitext(file_name)[0]
                if re.match(pattern, file_base):
                    sequence_files.append(file_path)
            
            # Sort files naturally (by number, not alphabetically)
            def natural_sort_key(path):
                base_name = os.path.splitext(os.path.basename(path))[0]
                match = re.match(pattern, base_name)
                if match:
                    return (match.group(1), int(match.group(2)))
                return (base_name, 0)
            
            sequence_files.sort(key=natural_sort_key)
            
            # Consider it a sequence if we have more than 1 file
            if len(sequence_files) > 1:
                return True, sequence_files
    
    return False, []

def load_image_sequence(image_path, parent_window):
    """
    Load an image or image sequence. Detects sequences automatically and prompts user.
    """
    # Check if it's part of a sequence
    is_sequence, sequence_files = detect_image_sequence(image_path)   
    files_to_load = [image_path]  # Default to single image
    
    if is_sequence:
        # Prompt user to choose between single image or sequence
        msg_box = QMessageBox(parent_window)
        msg_box.setWindowTitle("Image Sequence Detected")
        msg_box.setText(f"The selected image appears to be part of a sequence with {len(sequence_files)} images.")
        msg_box.setInformativeText("Would you like to load the entire sequence or just the single image?")
        
        sequence_button = msg_box.addButton("Load Sequence", QMessageBox.AcceptRole)
        single_button = msg_box.addButton("Load Single Image", QMessageBox.RejectRole)
        cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        
        msg_box.exec()
        
        if msg_box.clickedButton() == sequence_button:
            files_to_load = sequence_files
        elif msg_box.clickedButton() == single_button:
            files_to_load = [image_path]
        else:  # Cancel
            return 0
    
    # Create temp directories
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(frames_dir)
    os.makedirs(mask_dir)
    os.makedirs(matting_dir)
    
    print(f"Loading {'image sequence' if len(files_to_load) > 1 else 'image'}: {len(files_to_load)} file(s)")
    
    # Create progress dialog
    progress_dialog = QProgressDialog("Loading images...", "Cancel", 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()
    
    # Get settings
    settings_mgr = get_settings_manager()
    app_frame_format = settings_mgr.get_app_setting("frame_format", "png")
    
    # Load first image to get dimensions
    first_image = cv2.imread(files_to_load[0])
    if first_image is None:
        progress_dialog.close()
        QMessageBox.critical(parent_window, "Error", f"Could not load image: {files_to_load[0]}")
        return 0
    
    VideoInfo.height, VideoInfo.width = first_image.shape[:2]
    VideoInfo.fps = 24.0  # Default FPS for image sequences
    VideoInfo.total_frames = len(files_to_load)
    
    # Process each image
    for frame_count, source_path in enumerate(files_to_load):
        # Load image
        image = cv2.imread(source_path)
        if image is None:
            print(f"Warning: Could not load {source_path}, skipping...")
            continue
        
        # Determine output format
        source_ext = os.path.splitext(source_path)[1].lower()
        if source_ext in ['.png', '.jpg', '.jpeg']:
            # For PNG/JPG, copy as-is regardless of frame_format setting
            output_ext = source_ext.lstrip('.')
            frame_filename = os.path.join(frames_dir, f"{frame_count:05d}.{output_ext}")
            shutil.copy2(source_path, frame_filename)
        else:
            # For other formats, convert to the specified frame_format
            frame_filename = os.path.join(frames_dir, f"{frame_count:05d}.{app_frame_format}")
            cv2.imwrite(frame_filename, image)
        
        # Update progress
        progress_dialog.setValue((frame_count + 1) * 100 // len(files_to_load))
        QApplication.processEvents()
        
        # Check for cancellation
        if progress_dialog.wasCanceled():
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            progress_dialog.close()
            return 0
    
    progress_dialog.setValue(100)
    return VideoInfo.total_frames
    

# resume previous session
def resume_session():
    if os.path.exists(temp_dir):
        if os.path.exists(frames_dir) and os.listdir(frames_dir): # if there are frames
            print("Resuming previous session...")
            restore_video_info()
            return VideoInfo.total_frames
            
# Function to count the number of frames, used when launching the application to resume previous work
def restore_video_info():
    if not os.path.exists(frames_dir):
        return 0
    image = load_base_frame(0)
    if image is not None:
        height, width, channels = image.shape
        VideoInfo.width = width
        VideoInfo.height = height
        VideoInfo.fps = 24  # Default fallback
        extension = get_frame_extension()
        VideoInfo.total_frames = len([f for f in os.listdir(frames_dir) if f.endswith(f".{extension}")])
        
        # Try to load from session settings if available
        settings_mgr = get_settings_manager()
        if settings_mgr.session_exists():
            # Update VideoInfo with session data if available
            session_width = settings_mgr.get_session_setting("video_width", 0)
            session_height = settings_mgr.get_session_setting("video_height", 0)
            session_fps = settings_mgr.get_session_setting("video_fps", 0)
            session_frames = settings_mgr.get_session_setting("total_frames", 0)
            
            if session_width > 0 and session_height > 0:
                VideoInfo.width = session_width
                VideoInfo.height = session_height
            if session_fps > 0:
                VideoInfo.fps = session_fps
            if session_frames > 0:
                VideoInfo.total_frames = session_frames


# Load the smoothing model for antialiasing
def load_smoothing_model():
    global smoothing_model
    if smoothing_model is None:
        device = DeviceManager.get_device()
        try:
            smoothing_model = prepare_smoothing_model("./checkpoints/1x_binary_mask_smooth.pth", device)
        except Exception as e:
            print(f"Warning: Could not load antialiasing model: {e}")
            smoothing_model = None

def load_project(file_name, parent_window):
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    progress = None

    try:
        # Open archive to get file count
        with zipfile.ZipFile(file_name, 'r') as zipf:
            file_list = zipf.namelist()
            total_files = len(file_list)
        
        if total_files == 0:
            return True  # Empty archive, nothing to extract
        
        # Create progress dialog
        progress = QProgressDialog("Extracting files...", "", 0, total_files, parent_window)
        progress.setWindowTitle("Extracting Backup")
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.show()
        
        # Process the application events to ensure the dialog shows
        QApplication.processEvents()

        # Extract files
        with zipfile.ZipFile(file_name, 'r') as zipf:
            for i, file_name in enumerate(file_list):
                
                # Update progress dialog
                progress.setValue(i)
                progress.setLabelText(f"Extracting: {os.path.basename(file_name)}")
                QApplication.processEvents()
                
                # Extract file
                zipf.extract(file_name, temp_dir)
        
        # Complete the progress
        progress.setValue(total_files)
        progress.setLabelText("Extraction completed!")
        QApplication.processEvents()
        
        return True
        
    except Exception as e:
        if progress:
            progress.close()
        raise e
    
    finally:
        # Close progress dialog
        if progress:
            progress.close()


def save_project(file_name, parent_window):
    # Count total files for progress tracking
    total_files = 0
    all_files = []
    
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, temp_dir)
            all_files.append((file_path, arcname))
            total_files += 1

    # Create progress dialog
    progress = QProgressDialog("Backing up files...", "Cancel", 0, total_files, parent_window)
    progress.setWindowTitle("Creating Backup")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)  # Show immediately
    progress.show()
    
    # Process the application events to ensure the dialog shows
    QApplication.processEvents()

    try:
        # Create the backup archive
        with zipfile.ZipFile(file_name, 'w', zipfile.ZIP_STORED) as zipf:
            for i, (file_path, arcname) in enumerate(all_files):
                # Check if user cancelled
                if progress.wasCanceled():
                    # Remove partially created file
                    try:
                        os.remove(file_name)
                    except:
                        pass
                    return 0
                
                # Update progress dialog
                progress.setValue(i)
                progress.setLabelText(f"Adding: {os.path.basename(file_path)}")
                QApplication.processEvents()
                
                # Add file to archive
                zipf.write(file_path, arcname)
        
        # Complete the progress
        progress.setValue(total_files)
        #progress.setLabelText("Completed!")
        QApplication.processEvents()
        
    except Exception as e:
        progress.close()
        # Remove partially created file on error
        try:
            os.remove(file_name)
        except:
            pass
        raise e
    
    finally:
        # Close progress dialog
        if not progress.wasCanceled():
            progress.close()
    return 1
