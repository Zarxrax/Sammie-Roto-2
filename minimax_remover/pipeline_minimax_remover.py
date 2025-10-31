from typing import List, Optional, Union
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from PySide6.QtWidgets import QApplication
from minimax_remover.transformer_minimax_remover import Transformer3DModel

def clear_device_cache(device):
    """Clear cache for the appropriate device"""
    if device.type == 'mps':
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    elif device.type == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Minimax_Remover_Pipeline(DiffusionPipeline):

    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        
        self.manual_offload = True
        self.offload_device = torch.device("cpu")

    def enable_manual_offloading(self):
        """Enable manual CPU offloading for better memory management"""
        self.manual_offload = True
        
    def enable_vae_slicing(self):
        """Enable sliced VAE decoding for lower memory usage"""
        if hasattr(self.vae, 'enable_slicing'):
            self.vae.enable_slicing()
            #print("VAE slicing enabled")
        
    def enable_vae_tiling(self):
        """Enable tiled VAE decoding for lower memory usage"""
        if hasattr(self.vae, 'enable_tiling'):
            self.vae.enable_tiling()
    
    def offload_vae_to_cpu(self, device):
        """Move VAE to CPU and clear cache"""
        if self.manual_offload:
            self.vae.to(self.offload_device)
            clear_device_cache(device)
    
    def offload_transformer_to_cpu(self, device):
        """Move transformer to CPU and clear cache"""
        if self.manual_offload:
            self.transformer.to(self.offload_device)
            clear_device_cache(device)
    
    def load_vae_to_gpu(self, device):
        """Move VAE to GPU"""
        if self.manual_offload:
            self.vae.to(device)
    
    def load_transformer_to_gpu(self, device):
        """Move transformer to GPU"""
        if self.manual_offload:
            self.transformer.to(device)


    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 720,
        width: int = 1280,
        num_latent_frames: int = 21,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def progress_dialog(self):
        """Get the current progress dialog for cancellation checks"""
        return getattr(self, '_progress_dialog', None)

    @progress_dialog.setter
    def progress_dialog(self, dialog):
        """Set the progress dialog for cancellation checks"""
        self._progress_dialog = dialog
        
    @torch.no_grad()
    def __call__(
        self,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        images: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np"
    ):

        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        batch_size = 1
        transformer_dtype = torch.float16
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        num_channels_latents = 16

        # Prepare masks and images first
        masks = masks.repeat(1, 1, 1, 3) # Expand to 3 channels: (f, h, w) -> (f, h, w, 3)
        masks = masks.permute(3, 0, 1, 2) # Rearrange: (f, h, w, c) -> (c, f, h, w)
        masks = masks[None,...] # Add batch dimension: (c, f, h, w) -> (1, c, f, h, w)
        masks[masks>0] = 1 # make sure masks are binary
        images = images.permute(3, 0, 1, 2) # Rearrange: (f, h, w, c) -> (c, f, h, w)
        masked_images = images * (1-masks)

        # Create latents_mean and latents_std
        latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(torch.float16)
            )

        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                torch.float16
            )

        # PHASE 1: VAE ENCODING
        print("Phase 1: VAE Encoding...")
        if getattr(self, "_interrupt", False):
            raise RuntimeError("Inference cancelled by user")
        if self.progress_dialog is not None:
            self.progress_dialog.setLabelText("Phase 1: VAE Encoding...")
            QApplication.processEvents()
            if self.progress_dialog.wasCanceled():
                print("Inference cancelled during VAE encoding")
                raise RuntimeError("Inference cancelled by user")
            
        QApplication.processEvents()
        self.load_vae_to_gpu(device)

        # Encode to get the ACTUAL latent shape
        masked_latents = self.vae.encode(masked_images.half()).latent_dist.mode()
        masks_latents = self.vae.encode(2*masks.half()-1.0).latent_dist.mode()
        
        # Get actual latent dimensions from encoding
        actual_latent_frames = masked_latents.shape[2]
        
        # Move latents_mean and latents_std to the same device as masked_latents
        latents_mean = latents_mean.to(masked_latents.device)
        latents_std = latents_std.to(masked_latents.device)

        masked_latents = (masked_latents - latents_mean) * latents_std
        masks_latents = (masks_latents - latents_mean) * latents_std
        
        # Create noise latents with the CORRECT shape from actual encoding
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            actual_latent_frames,  # Use actual encoded shape, not calculated
            torch.float16,
            device,
            generator,
            None,
        )
        #print(f"Prepared latents shape: {latents.shape}")

        # Free memory from encoding
        del masked_images, images, masks
        clear_device_cache(device)

        # Offload VAE, load Transformer
        self.offload_vae_to_cpu(device)
        self.load_transformer_to_gpu(device)

        # PHASE 2: TRANSFORMER DENOISING
        print("Phase 2: Transformer Denoising...")
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Check for cancellation
                if getattr(self, "_interrupt", False):
                    raise RuntimeError("Inference cancelled by user")
                if self.progress_dialog is not None:
                    QApplication.processEvents()
                    if self.progress_dialog.wasCanceled():
                        print(f"Inference cancelled at step {i+1}/{num_inference_steps}")
                        raise RuntimeError("Inference cancelled by user")
                    
                    # Update progress dialog with step information
                    self.progress_dialog.setLabelText(f"Phase 2: Transformer Denoising ({i}/{num_inference_steps})...")

                QApplication.processEvents()
                latent_model_input = latents.to(transformer_dtype)
                latent_model_input = torch.cat([latent_model_input, masked_latents, masks_latents], dim=1)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input.half(),
                    timestep=timestep
                )[0]

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                progress_bar.update()
                QApplication.processEvents()

        # Free transformer-related tensors
        del masked_latents, masks_latents, noise_pred, latent_model_input
        clear_device_cache(device)
        
        # Offload Transformer, load VAE
        self.offload_transformer_to_cpu(device)
        self.load_vae_to_gpu(device)

        # PHASE 3: VAE DECODING
        print("Phase 3: VAE Decoding...")
        if getattr(self, "_interrupt", False):
            raise RuntimeError("Inference cancelled by user")
        if self.progress_dialog is not None:
            self.progress_dialog.setLabelText("Phase 3: VAE Decoding...")
            QApplication.processEvents()
            if self.progress_dialog.wasCanceled():
                print("Inference cancelled during VAE decoding")
                raise RuntimeError("Inference cancelled by user")
        QApplication.processEvents()

        # Move latents_mean and latents_std to the same device as latents for decoding
        latents_mean = latents_mean.to(latents.device)
        latents_std = latents_std.to(latents.device)
        
        latents = latents.half() / latents_std + latents_mean

        video = self.vae.decode(latents, return_dict=False)[0]

        video = self.video_processor.postprocess_video(video, output_type=output_type)

        clear_device_cache(device)

        return WanPipelineOutput(frames=video)
