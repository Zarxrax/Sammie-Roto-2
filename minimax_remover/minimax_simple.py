import os
import cv2
import numpy as np
import torch
import argparse
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

def load_model(model_path="./checkpoints/", device="cuda"):
    """Load the minimax remover pipeline."""
    vae = AutoencoderKLWan.from_pretrained(
        f"{model_path}/vae", 
        torch_dtype = torch.float16
    )
    transformer = Transformer3DModel.from_pretrained(
        f"{model_path}/transformer", 
        torch_dtype = torch.float16
    )
    scheduler = UniPCMultistepScheduler.from_pretrained(
        f"{model_path}/scheduler"
    )
    
    pipe = Minimax_Remover_Pipeline(
        transformer=transformer, 
        vae=vae, 
        scheduler=scheduler
    )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
        pipe.enable_vae_slicing()
        #pipe.enable_vae_tiling()
    
    return pipe

def load_frames(image_folder, mask_folder):
    """Load image frames and corresponding masks."""
    image_files = sorted([f for f in os.listdir(image_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    images = []
    masks = []
    
    for img_file in image_files:
        # Load image
        img_path = os.path.join(image_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        
        # Extract frame number from filename (assuming format like 00001.png)
        base_name = os.path.splitext(img_file)[0]
        
        # Look for mask in subfolder: mask_folder/00001/0.png
        mask_path = os.path.join(mask_folder, base_name, '0.png')
        
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found at {mask_path}")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        
        masks.append(mask)
    
    return images, masks, image_files

def preprocess_batch(images, masks, width, height, device="cuda"):
    """Preprocess images and masks for the model."""
    out_images = []
    out_masks = []
    
    for img, msk in zip(images, masks):
        # Resize image to target resolution
        img_resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        out_images.append(img_resized.astype(np.float32) / 127.5 - 1.0)
        
        # Resize mask to target resolution
        msk_resized = cv2.resize(msk, (width, height), interpolation=cv2.INTER_NEAREST)
        msk_resized = (msk_resized.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        out_masks.append(msk_resized)
    
    arr_images = torch.from_numpy(np.stack(out_images)).half().to(device)
    arr_masks = torch.from_numpy(np.stack(out_masks)).half().to(device)
    
    return arr_images, arr_masks

def process_frames(pipe, images, masks, output_folder, image_files, 
                   width, height, dilation_iterations=6, num_inference_steps=6, 
                   random_seed=42, device="cuda"):
    """Process frames through the removal pipeline."""
    os.makedirs(output_folder, exist_ok=True)
    
    # Pad frames to be divisible by VAE temporal scale factor (4)+1
    num_input_frames = len(images)
    vae_temporal_scale = 4
    pad_frames = (vae_temporal_scale - (num_input_frames % vae_temporal_scale)) % vae_temporal_scale + 1
    
    if pad_frames > 0:
        # Replicate last frame for padding
        for _ in range(pad_frames):
            images.append(images[-1].copy())
            masks.append(masks[-1].copy())

    # Preprocess
    img_tensor, mask_tensor = preprocess_batch(images, masks, width, height, device)
    mask_tensor = mask_tensor[:, :, :, None]
    
    # Run inference
    print(f"Processing {len(images)-pad_frames} frames at {width}x{height} resolution...")
    with torch.no_grad():
        output = pipe(
            images=img_tensor,
            masks=mask_tensor,
            num_frames=mask_tensor.shape[0],
            height=height,
            width=width,
            num_inference_steps=int(num_inference_steps),
            generator=torch.Generator(device=device).manual_seed(random_seed),
            iterations=int(dilation_iterations)
        ).frames[0]
    
    # Remove padded frames from output
    if pad_frames > 0:
        output = output[:num_input_frames]

    output = np.uint8(output * 255)
    
    # Save output frames
    print(f"Saving frames to {output_folder}...")
    for i, (frame, filename) in enumerate(zip(output, image_files)):
        output_path = os.path.join(output_folder, filename)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, frame_bgr)
        if (i + 1) % 10 == 0:
            print(f"Saved {i + 1}/{len(output)} frames")
    
    print("Processing complete!")

def main():
    parser = argparse.ArgumentParser(description='Process image frames with mask removal')
    parser.add_argument('--image_folder', type=str, default='./input_frames/',
                       help='Path to folder containing input images')
    parser.add_argument('--mask_folder', type=str, default='./mask_frames/',
                       help='Path to folder containing mask images')
    parser.add_argument('--output_folder', type=str, default='./output_frames/',
                       help='Path to folder for output images')
    parser.add_argument('--model_path', type=str, default='./model/',
                       help='Path to model directory')
    parser.add_argument('--width', type=int, default=832,
                       help='Output width in pixels')
    parser.add_argument('--height', type=int, default=480,
                       help='Output height in pixels')
    parser.add_argument('--dilation', type=int, default=6,
                       help='Mask dilation iterations')
    parser.add_argument('--steps', type=int, default=6,
                       help='Number of inference steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.device == "cuda":
        print("CUDA not available, using CPU")
    
    print("Loading model...")
    pipe = load_model(args.model_path, device)
    
    print("Loading frames...")
    images, masks, image_files = load_frames(args.image_folder, args.mask_folder)
    print(f"Loaded {len(images)} frames")
    
    print("Processing frames...")
    process_frames(
        pipe, images, masks, args.output_folder, image_files,
        width=args.width,
        height=args.height,
        dilation_iterations=args.dilation,
        num_inference_steps=args.steps,
        random_seed=args.seed,
        device=device
    )

if __name__ == "__main__":
    main()
