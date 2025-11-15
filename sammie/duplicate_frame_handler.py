import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QProgressDialog, QApplication
from sammie.settings_manager import get_settings_manager

# Resolve absolute path of file back to project root folder
utils_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(utils_dir, ".."))

frames_dir = os.path.join(project_root, "temp", "frames")
mask_dir = os.path.join(project_root, "temp", "masks")
backup_dir = os.path.join(project_root, "temp", "masks_backup")

# Use ORB comparison from opencv to compare two input frames/images for similarity
def orb_comparison(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Check for None descriptors (no features found)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return 0.0
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    similarity_score = len(matches) / max(len(kp1), len(kp2))
    return similarity_score

# To compare without other elements or the background on the frame affecting the comparison, the mask luma matte gets applied to the frame
def generate_matted_frame(frame_path, mask_dir, frame_number):
    frame = cv2.imread(frame_path)
    frame_mask_dir = os.path.join(mask_dir, frame_number)
    if not os.path.exists(frame_mask_dir):
        # Warning could not find masks
        print(f"Missing masks folder for frame: {frame_number}")
        return None
    
    # Create empty mask image
    mask_image = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    # Combine all masks from mask folder into one mask image for comparison
    for matte in os.listdir(frame_mask_dir):
        matte_path = os.path.join(frame_mask_dir, matte)
        matte_image = cv2.imread(matte_path, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.bitwise_or(mask_image, matte_image)

    # Check if there is no luma mask for a frame, use the full frame in that case to prevent zero ORB matches (which raised a division by zero error), but does result in zero similarity.
    if cv2.countNonZero(mask_image) == 0:
        mask_image = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
    # Use the combined mask image as the overall mask luma matte
    result_frame = cv2.bitwise_and(frame, frame, mask=mask_image)
    return result_frame

# Replace the masks on disc with a specific "similar frames" list
def replace_files_similar_mattes(mask_dir, similar_frames):
    last_mask_dir = os.path.join(mask_dir, similar_frames[-1])
    file_list = os.listdir(last_mask_dir)
    for i, frame in enumerate(similar_frames):
        if i == len(similar_frames) - 1:  # Skip the last sourcing frame
            break
        for file in file_list:
            file_path = os.path.join(last_mask_dir, file)
            replace_mask_dir = os.path.join(mask_dir, frame, file)
            shutil.copy(file_path, replace_mask_dir)

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
    # Get the list of propagated frame numbers
    for filename in os.listdir(frames_dir):
        if filename.endswith(".png"):
            frame_numbers.append(os.path.splitext(filename)[0])
            
    # Check if the masks directory has the same amount of masks as the amount of frames in the video file
    num_of_masks = len(os.listdir(mask_dir))
    frames_amount = len(frame_numbers)
    if num_of_masks != frames_amount:
        print("Mismatch between frames and masks.\nPlease fully track objects first.")
        return False
    
    # If a backup of the masks exists, it means deduplication has been executed before
    if os.path.exists(backup_dir): # Restore the original masks to ensure we're working with the source data.
        restore_backup_mattes(mask_dir, backup_dir)
    else: # Create a backup of the original masks
        backup_mattes(mask_dir, backup_dir)

    frame_index = 0 # Keeps track of the current "base" frame for comparisons
    deduped_frames_amount = 0 # Keep track of how many frames/masks have been replaced/deduped

    # Initialize the base frame
    start_base_frame_path = os.path.join(frames_dir, frame_numbers[frame_index] + ".png")
    base_frame = generate_matted_frame(start_base_frame_path, mask_dir, frame_numbers[frame_index]) # Frame used for ORB comparison

    progress = tqdm(total=frames_amount, desc="Deduplicating mask frames...")
    progress_dialog = QProgressDialog("Deduplicating...", "", 0, 100, parent_window)
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
            next_frame_path = os.path.join(frames_dir, frame_numbers[next_index] + ".png")
            next_frame = generate_matted_frame(next_frame_path, mask_dir, frame_numbers[next_index])
            
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
            # Load the next frame
            new_base_frame_path = os.path.join(frames_dir, frame_numbers[frame_index] + ".png")
            base_frame = generate_matted_frame(new_base_frame_path, mask_dir, frame_numbers[frame_index])