import os
import torch
from torch.cuda.amp import autocast
from torchvision import transforms
import cv2
import numpy as np
from sammie.srvgg_arch import SRVGGNetCompact

# Load and prepare the model
def prepare_smoothing_model(weights_path, device):
    model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=24, num_conv=8, upscale=1, act_type='prelu')
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

# Preprocess the input image
def preprocess_image(input_mask, device):
    mask_normalized = input_mask.astype(np.float32) / 255.0
    mask_tensor = torch.from_numpy(np.transpose(mask_normalized, (2, 0, 1))).unsqueeze(0).to(device)
    return mask_tensor

# Postprocess and save the output image
def postprocess(output_tensor):
    # Remove the batch dimension and convert to NumPy
    output_array = output_tensor.squeeze(0).cpu().numpy()
    # Clamp values to [0, 1] and scale to [0, 255]
    output_array = np.clip(output_array, 0, 1) * 255
    # Convert to uint8 format
    output_array = output_array.astype(np.uint8)
    # Rearrange dimensions from CHW to HWC for image representation
    output_array = np.transpose(output_array, (1, 2, 0))
    return output_array

# Process a single image
def run_smoothing_model(input_mask, model, device):
    input_tensor = preprocess_image(input_mask, device)
    with torch.no_grad():
        if device.type == "cuda":
            # Enable automatic mixed precision
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output_tensor = model(input_tensor)
        else:
            output_tensor = model(input_tensor)
    return postprocess(output_tensor)

