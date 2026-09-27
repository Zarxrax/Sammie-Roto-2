"""MiniMax-Remover checkpoints."""
from sammie.model_downloader import DownloadSpec

DOWNLOADS = {
    "minimax_transformer": DownloadSpec("https://huggingface.co/zibojia/minimax-remover/resolve/main/transformer/diffusion_pytorch_model.safetensors", "183c7a631e831f73f8da64c5c4d83e2f", "checkpoints/minimax/transformer"),
    "minimax_vae": DownloadSpec("https://huggingface.co/zibojia/minimax-remover/resolve/main/vae/diffusion_pytorch_model.safetensors", "3f80444947443d8f36c0ed2497c20c8d", "checkpoints/minimax/vae"),
}
