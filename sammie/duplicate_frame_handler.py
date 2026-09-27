from sammie import image_ops
import numpy as np
import os
import shutil
from tqdm import tqdm
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication
from sammie.settings_manager import get_settings_manager
from sammie import core

# Resolve absolute path of file back to project root folder
utils_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(utils_dir, ".."))

frames_dir = os.path.join(project_root, "temp", "frames")
mask_dir = os.path.join(project_root, "temp", "masks")
backup_dir = os.path.join(project_root, "temp", "masks_backup")

# Compare the isolated object appearance at a small common resolution.
def orb_comparison(img1, img2):
    if img1 is None or img2 is None:
        return 0.0
    a = image_ops.resize(img1, (64, 64), interpolation=image_ops.INTER_AREA).astype(np.float32)
    b = image_ops.resize(img2, (64, 64), interpolation=image_ops.INTER_AREA).astype(np.float32)
    foreground = np.any(a > 0, axis=2) | np.any(b > 0, axis=2)
    if not np.any(foreground):
        return 0.0
    difference = np.mean(np.abs(a[foreground] - b[foreground])) / 255.0
    return float(np.clip(1.0 - difference, 0.0, 1.0))

# To compare without other elements or the background on the frame affecting the comparison, the mask luma matte gets applied to the frame
def generate_matted_frame(frame_path, mask_dir, frame_number):
    frame = image_ops.imread(frame_path)
    object_ids = set(core.output_ids(mask_dir, frame_number))
    if core.paint_enabled:
        object_ids.update(core.output_ids(core.paint_dir, frame_number))
    if not object_ids:
        # Return None if masks are missing for this frame
        print(f"Missing masks for frame: {frame_number}")
        return None
    
    # Create empty mask image
    mask_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    # Combine all masks from mask folder into one mask image for comparison
    for object_id in object_ids:
        matte_image = core.load_segmentation_mask(frame_number, object_id)
        if matte_image is None:
            continue
        mask_image = image_ops.bitwise_or(mask_image, matte_image)

    # Check if there is no luma mask for a frame, use the full frame in that case to prevent zero ORB matches (which raised a division by zero error), but does result in zero similarity.
    if image_ops.countNonZero(mask_image) == 0:
        mask_image = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
    # Use the combined mask image as the overall mask luma matte
    result_frame = image_ops.bitwise_and(frame, frame, mask=mask_image)
    return result_frame

# Replace the masks on disc with a specific "similar frames" list
def replace_files_similar_mattes(mask_dir, similar_frames):
    last_frame = similar_frames[-1]
    object_ids = core.output_ids(mask_dir, last_frame)
    
    # Check if the source mask directory exists
    if not object_ids:
        print(f"Source masks missing for frame: {last_frame}, skipping replacement")
        return
    
    for i, frame in enumerate(similar_frames):
        if i == len(similar_frames) - 1:  # Skip the last sourcing frame
            break
        
        # Check if the target mask directory exists before attempting to replace
        if not core.output_ids(mask_dir, frame):
            print(f"Target masks missing for frame: {frame}, skipping")
            continue
            
        for object_id in object_ids:
            shutil.copy(core.output_path(mask_dir, last_frame, object_id),
                        core.output_path(mask_dir, frame, object_id))

def backup_mattes(mask_dir, backup_dir):
    #print("Creating original masks backup")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(mask_dir, backup_dir)

def restore_backup_mattes(mask_dir, backup_dir):
    if not os.path.exists(backup_dir):
        print("No backup masks folder was found.")
        return
    print("Restoring original masks backup")
    shutil.copytree(backup_dir, mask_dir, dirs_exist_ok=True)

def remove_backup_mattes():
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

# Main function to replace similar (matted) frames with one single matte frame
def replace_similar_matte_frames(parent_window, dedupe_min_threshold):
    settings_mgr = get_settings_manager()

    frame_numbers = []

    # Check if the frames directory exists
    if not os.path.exists(frames_dir):
        print("Could not find frames to dedupe.\nPlease load a video first.")
        return False
    
    # Get the indexed frames; source plate numbers may start at any value.
    extension = get_settings_manager().get_session_setting("frame_format", "png")
    frame_numbers = list(range(core.VideoInfo.total_frames))
    
    # Check if masks directory exists
    if not os.path.exists(mask_dir):
        print("No masks directory found.\nPlease track objects first.")
        return False
    
    # If a backup of the masks exists, it means deduplication has been executed before
    if os.path.exists(backup_dir): # Restore the original masks to ensure we're working with the source data.
        restore_backup_mattes(mask_dir, backup_dir)
    else: # Create a backup of the original masks
        backup_mattes(mask_dir, backup_dir)

    frame_index = 0 # Keeps track of the current "base" frame for comparisons
    deduped_frames_amount = 0 # Keep track of how many frames/masks have been replaced/deduped
    frames_amount = len(frame_numbers)

    # Find the first frame with valid masks to use as initial base frame
    base_frame = None
    while frame_index < frames_amount and base_frame is None:
        start_base_frame_path = core.frame_path(frame_numbers[frame_index], extension)
        base_frame = generate_matted_frame(start_base_frame_path, mask_dir, frame_numbers[frame_index])
        if base_frame is None:
            frame_index += 1

    # If no frames have masks at all, exit
    if base_frame is None:
        print("No valid masks found in any frames.")
        return False

    progress = tqdm(total=frames_amount, desc="Deduplicating mask frames...")
    progress_dialog = QProgressDialog("Deduplicating...", None, 0, 100, parent_window)
    progress_dialog.setWindowTitle("Progress")
    progress_dialog.setWindowModality(Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.show()
    
    while True:
        similar_frames = []
        similar_frames.append(frame_numbers[frame_index])

        for next_index in range(frame_index + 1, len(frame_numbers)):
            # Update progress dialog
            progress_dialog.setValue((next_index)*100/(frames_amount))
            QApplication.processEvents()
            progress.update(1)
            
            # Load the next frame
            next_frame_path = core.frame_path(frame_numbers[next_index], extension)
            next_frame = generate_matted_frame(next_frame_path, mask_dir, frame_numbers[next_index])
            
            # If the next frame has no masks, treat it as a break point
            if next_frame is None:
                break
            
            # Compare the current frame with the next frame
            similarity_score = orb_comparison(base_frame, next_frame)
            if similarity_score > dedupe_min_threshold:
                # If the frames are similar enough, add the next checked frame to the similar_frames list
                similar_frames.append(frame_numbers[next_index])
            else:
                # If the frames are not similar, break out of the inner loop
                break
        
        replace_files_similar_mattes(mask_dir, similar_frames)
        
        # Find the actual index of the last similar frame in the input list and update the frame_index from that point onwards
        last_similar_frame_index = frame_numbers.index(similar_frames[-1])
        frame_index = last_similar_frame_index + 1

        # Update the amount of deduped frames
        deduped_frames_amount += (len(similar_frames)-1) # Base frame gets stored in the list as well, hence the subtraction

        # Check if all the frames have been processed
        if frame_index >= frames_amount:
            # Force complete progress bar and display info
            progress_dialog.setValue(100)
            progress.n = frames_amount
            progress.refresh()
            progress.close()
            print(f"Deduplicated {deduped_frames_amount} mask frames")
            settings_mgr.set_session_setting("is_deduplicated", True)
            return True
        else:
            # Find the next frame with valid masks
            base_frame = None
            while frame_index < frames_amount and base_frame is None:
                new_base_frame_path = core.frame_path(frame_numbers[frame_index], extension)
                base_frame = generate_matted_frame(new_base_frame_path, mask_dir, frame_numbers[frame_index])
                if base_frame is None:
                    # Skip frames without masks
                    frame_index += 1
            
            # If we've run out of frames with masks, we're done
            if base_frame is None:
                progress_dialog.setValue(100)
                progress.n = frames_amount
                progress.refresh()
                progress.close()
                print(f"Deduplicated {deduped_frames_amount} mask frames")
                settings_mgr.set_session_setting("is_deduplicated", True)
                return True
