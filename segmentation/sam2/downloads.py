"""SAM2 and EfficientTAM checkpoint declarations."""
from sammie.model_downloader import DownloadSpec

DOWNLOADS = {
    "Large": DownloadSpec("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt", "2b30654b6112c42a115563c638d238d9", "checkpoints"),
    "Base": DownloadSpec("https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt", "ec7bd7d23d280d5e3cfa45984c02eda5", "checkpoints"),
    "Efficient": DownloadSpec("https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt", "962e151a9dca3b75d8228a16e5264010", "checkpoints"),
}
