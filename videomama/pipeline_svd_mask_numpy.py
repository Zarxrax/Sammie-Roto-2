# pipeline_svd_mask_numpy.py

import numpy as np
import torch

from PySide6.QtWidgets import QApplication

from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel


class VideoInferencePipeline:
    """
    A reusable pipeline for single-step video diffusion inference.

    This class encapsulates the models and the core inference logic,
    separating it from data loading and saving, which can vary between tasks.
    """

    def __init__(self, base_model_path: str, unet_checkpoint_path: str, device: str = "cuda",
                 weight_dtype: torch.dtype = torch.float16, enable_model_cpu_offload: bool = False,
                 vae_encode_chunk_size: int = 8, attention_mode: str = "auto",
                 enable_vae_tiling: bool = False, enable_vae_slicing: bool = True):
        """
        Loads all necessary models into memory.

        Args:
            base_model_path (str): Path to the base Stable Video Diffusion model.
            unet_checkpoint_path (str): Path to the fine-tuned UNet checkpoint.
            device (str): The device to run models on ('cuda' or 'cpu').
            weight_dtype (torch.dtype): The precision for model weights (float16 or bfloat16).
            enable_model_cpu_offload (bool): If True, models are kept on CPU and moved to GPU only when needed.
            vae_encode_chunk_size (int): Number of frames to encode at once in VAE (lower = less memory).
            attention_mode (str): Attention optimization: 'auto', 'xformers', 'sdpa', or 'none'.
            enable_vae_tiling (bool): Enable tiled VAE encoding/decoding for lower memory.
            enable_vae_slicing (bool): Enable VAE slicing to process one image at a time.
        """
        #print("--- Initializing Inference Pipeline and Loading Models ---")
        self.device = torch.device(device)
        self.weight_dtype = weight_dtype
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.vae_encode_chunk_size = vae_encode_chunk_size

        # Load models from pretrained paths
        try:
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(base_model_path, subfolder="vae", variant="fp16")
            self.unet = UNetSpatioTemporalConditionModel.from_pretrained(unet_checkpoint_path, subfolder="unet")
        except Exception as e:
            raise IOError(f"Fatal error loading models: {e}")

        # Set models to evaluation mode
        self.vae.eval()
        self.unet.eval()

        # --- Apply attention optimizations ---
        self._apply_attention_optimization(attention_mode)

        # --- Apply VAE memory optimizations ---
        if enable_vae_slicing:
            try:
                self.vae.enable_slicing()
                #print("--- VAE Slicing ENABLED ---")
            except (AttributeError, NotImplementedError):
                #print("--- VAE Slicing not supported by this VAE version ---")
                pass

        if enable_vae_tiling:
            try:
                self.vae.enable_tiling()
                #print("--- VAE Tiling ENABLED ---")
            except (AttributeError, NotImplementedError):
                #print("--- VAE Tiling not supported by this VAE version ---")
                pass

        if self.enable_model_cpu_offload:
            # Keep models on CPU initially, move to GPU only when needed
            #print(f"--- Model CPU Offloading ENABLED (memory optimization) ---")
            self.vae.to("cpu", dtype=self.weight_dtype)
            self.unet.to("cpu", dtype=self.weight_dtype)
        else:
            # Move all models to GPU
            self.vae.to(self.device, dtype=self.weight_dtype)
            self.unet.to(self.device, dtype=self.weight_dtype)

        #print(f"--- Models Loaded Successfully on {self.device} ---")
        #print(f"--- VAE Encode Chunk Size: {self.vae_encode_chunk_size} frames ---")

    def _apply_attention_optimization(self, attention_mode: str):
        """Apply memory-efficient attention to the UNet."""
        if attention_mode == "none":
            print("--- Attention optimization: DISABLED ---")
            return

        # Try xformers first if requested or auto
        if attention_mode in ("auto", "xformers"):
            try:
                import xformers  # noqa: F401
                self.unet.enable_xformers_memory_efficient_attention()
                print("--- Attention optimization: xformers ENABLED ---")
                return
            except (ImportError, ModuleNotFoundError):
                if attention_mode == "xformers":
                    print("--- WARNING: xformers not installed, falling back to default attention ---")
            except Exception as e:
                print(f"--- WARNING: xformers failed ({e}), trying alternatives ---")

        # Try PyTorch 2.0+ SDPA
        if attention_mode in ("auto", "sdpa"):
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                try:
                    from diffusers.models.attention_processor import AttnProcessor2_0
                    self.unet.set_attn_processor(AttnProcessor2_0())
                    #print("--- Attention optimization: PyTorch SDPA ENABLED ---")
                    return
                except Exception as e:
                    if attention_mode == "sdpa":
                        print(f"--- WARNING: SDPA setup failed ({e}) ---")
            elif attention_mode == "sdpa":
                print("--- WARNING: SDPA requires PyTorch 2.0+, falling back to default ---")

        if attention_mode == "auto":
            print("--- Attention optimization: using default attention ---")

    @staticmethod
    def _clear_device_cache(device):
        """Clear cache for the appropriate device."""
        if device.type == 'mps':
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        elif device.type == 'xpu':
            torch.xpu.empty_cache()
        elif device.type == 'cuda':
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self, cond_frames, mask_frames, seed=42, fps=7, motion_bucket_id=127,
            noise_aug_strength=0.0, pbar=None, progress_callback=None):
        """
        Runs the core inference process on a sequence of conditioning and mask frames.

        Args:
            cond_frames (list[np.ndarray]): List of RGB uint8 arrays for conditioning.
            mask_frames (list[np.ndarray]): List of RGB or grayscale uint8 arrays for the masks.
            seed (int): Random seed for generation.
            fps (int): Frames per second to condition the model with.
            motion_bucket_id (int): Motion bucket ID for conditioning.
            noise_aug_strength (float): Noise augmentation strength.
            pbar: Optional ComfyUI ProgressBar for progress tracking.
            progress_callback: Optional callable(step, total_steps, description) for UI updates.

        Returns:
            list[np.ndarray]: A list of the generated video frames as RGB uint8 arrays.
        """
        def _notify(step, desc):
            """Notify progress via both pbar (ComfyUI) and callback (Sammie)"""
            if pbar is not None:
                pbar.update(1)
            if progress_callback is not None:
                try:
                    progress_callback(step, 5, desc)
                except Exception:
                    pass

        # --- 1. Prepare Tensors ---
        cond_video_tensor = self._cv2_to_tensor(cond_frames).to(self.device)
        mask_video_tensor = self._cv2_to_tensor(mask_frames).to(self.device)

        if mask_video_tensor.shape[2] != 3:
            mask_video_tensor = mask_video_tensor.repeat(1, 1, 3, 1, 1)

        with torch.no_grad():
            # --- 2. Build zero image embeddings ---
            # CLIP conditioning is not used by this fine-tuned model. The UNet was trained
            # with zeroed encoder_hidden_states, so we construct the zero tensor directly
            # using the UNet's expected cross_attention_dim rather than running CLIP.
            hidden_size = self.unet.config.cross_attention_dim
            encoder_hidden_states = torch.zeros(
                (1, 1, hidden_size), dtype=self.weight_dtype, device=self.device
            )

            #_notify(1, "Image embedding complete")

            # --- 3. Prepare Latents ---
            _notify(2, "VAE encoding...")
            if self.enable_model_cpu_offload:
                self.vae.to(self.device)

            cond_latents = self._tensor_to_vae_latent(cond_video_tensor.to(self.weight_dtype))
            cond_latents = cond_latents / self.vae.config.scaling_factor
            QApplication.processEvents()

            mask_latents = self._tensor_to_vae_latent(mask_video_tensor.to(self.weight_dtype))
            mask_latents = mask_latents / self.vae.config.scaling_factor
            QApplication.processEvents()

            # Free raw pixel tensors - no longer needed after VAE encoding
            del cond_video_tensor, mask_video_tensor
            self._clear_device_cache(self.device)

            if self.enable_model_cpu_offload:
                self.vae.to("cpu")
                self._clear_device_cache(self.device)

            #_notify(2, "VAE encoding complete")

            # --- 4. Run UNet Single-Step Inference ---
            _notify(3, "UNet inference...")
            if self.enable_model_cpu_offload:
                self.unet.to(self.device)

            generator = torch.Generator(device=self.device).manual_seed(seed)
            noisy_latents = torch.randn(cond_latents.shape, generator=generator, device=self.device,
                                        dtype=self.weight_dtype)
            timesteps = torch.full((1,), 1.0, device=self.device, dtype=torch.long)
            added_time_ids = self._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, batch_size=1)

            unet_input = torch.cat([noisy_latents, cond_latents, mask_latents], dim=2)
            # Free intermediate latents before UNet forward pass
            del noisy_latents, cond_latents, mask_latents
            self._clear_device_cache(self.device)

            pred_latents = self.unet(unet_input, timesteps, encoder_hidden_states, added_time_ids=added_time_ids).sample
            QApplication.processEvents()

            del unet_input
            if self.enable_model_cpu_offload:
                self.unet.to("cpu")
            self._clear_device_cache(self.device)

            #_notify(3, "UNet inference complete")

            # --- 5. Decode Latents to Video Frames ---
            _notify(4, "VAE decoding...")
            if self.enable_model_cpu_offload:
                self.vae.to(self.device)

            pred_latents = (1 / self.vae.config.scaling_factor) * pred_latents.squeeze(0)

            frames = []
            # Process in chunks to avoid VRAM issues (lower = less memory, slower)
            decode_chunk_size = min(self.vae_encode_chunk_size, pred_latents.shape[0])
            for i in range(0, pred_latents.shape[0], decode_chunk_size):
                chunk = pred_latents[i: i + decode_chunk_size]
                decoded_chunk = self.vae.decode(chunk, num_frames=chunk.shape[0]).sample
                frames.append(decoded_chunk.cpu())  # Move decoded frames to CPU immediately
                QApplication.processEvents()
            del pred_latents
            if self.enable_model_cpu_offload:
                self.vae.to("cpu")
            self._clear_device_cache(self.device)

            #_notify(4, "VAE decoding complete")

            video_tensor = torch.cat(frames, dim=0)
            del frames
            video_tensor = (video_tensor / 2.0 + 0.5).clamp(0, 1).mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

            # Return a list of RGB uint8 frames
            video_tensor = video_tensor.float().cpu()
            output_frames = []
            for frame in video_tensor:
                # frame: (3, H, W), float [0, 1]
                arr = (frame.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                output_frames.append(arr)
            return output_frames

    def _cv2_to_tensor(self, frames: list) -> torch.Tensor:
        """Converts a list of RGB uint8 arrays (H, W, 3) or grayscale (H, W) arrays
        to a normalized float video tensor of shape (1, F, 3, H, W) in [-1, 1]."""
        tensors = []
        for f in frames:
            arr = np.asarray(f)
            if arr.ndim == 2:
                # Grayscale -> replicate to 3 channels
                arr = np.stack([arr] * 3, axis=-1)
            t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0  # (3, H, W) in [0, 1]
            tensors.append(t)
        video_tensor = torch.stack(tensors).unsqueeze(0)  # (1, F, 3, H, W)
        return video_tensor * 2.0 - 1.0  # normalize to [-1, 1]

    def _tensor_to_vae_latent(self, t: torch.Tensor):
        """Encodes a video tensor into the VAE's latent space with optional chunking."""
        batch_size, video_length = t.shape[0], t.shape[1]
        #t = rearrange(t, "b f c h w -> (b f) c h w")
        t = t.reshape(t.shape[0] * t.shape[1], *t.shape[2:])

        # Encode in chunks to reduce memory usage
        latents_list = []
        for i in range(0, t.shape[0], self.vae_encode_chunk_size):
            chunk = t[i: i + self.vae_encode_chunk_size]
            chunk_latents = self.vae.encode(chunk).latent_dist.sample()
            latents_list.append(chunk_latents)

        latents = torch.cat(latents_list, dim=0)
        #latents = rearrange(latents, "(b f) c h w -> b f c h w", b=batch_size, f=video_length)
        latents = latents.reshape(batch_size, video_length, *latents.shape[1:])
        return latents * self.vae.config.scaling_factor

    def _get_add_time_ids(self, fps, motion_bucket_id, noise_aug_strength, batch_size):
        """Creates the additional time IDs for conditioning the UNet."""
        add_time_ids_list = [fps, motion_bucket_id, noise_aug_strength]
        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids_list)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created.")
        add_time_ids = torch.tensor([add_time_ids_list], dtype=self.weight_dtype, device=self.device)
        return add_time_ids.repeat(batch_size, 1)
