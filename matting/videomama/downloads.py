"""Checkpoint downloads owned by the VideoMaMa engine."""

from sammie.model_downloader import DownloadSpec


DOWNLOADS = {
    "videomama": DownloadSpec(
        url="https://huggingface.co/SammyLim/VideoMaMa/resolve/main/unet/diffusion_pytorch_model.safetensors",
        md5="c8d457d4d5eb90f274bd441df60c8e47",
        dest_dir="checkpoints/videomama/unet",
    ),
    "svd_vae": DownloadSpec(
        url="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors",
        md5="46a0af9a794fb405221988a7e2b1396b",
        dest_dir="checkpoints/videomama/vae",
    ),
}
