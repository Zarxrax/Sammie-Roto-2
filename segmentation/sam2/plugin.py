"""SAM2 and EfficientTAM model variants."""
from segmentation.registry import EngineSpec
from segmentation.sam2.downloads import DOWNLOADS


def _loader(model_id, config, checkpoint):
    def load(device, parent_window):
        from sammie.model_downloader import ensure_models
        from segmentation.sam2.vendor.build_sam import build_sam2_video_predictor
        if not ensure_models(DOWNLOADS[model_id], parent=parent_window):
            return None
        print(f"Loaded {model_id} segmentation model")
        return build_sam2_video_predictor(config, checkpoint, device=device)
    return load


ENGINES = (
    EngineSpec("Base", "Base", _loader("Base", "sam2.1_hiera_b+.yaml", "./checkpoints/sam2.1_hiera_base_plus.pt"),
               "Balanced SAM2 model.", 10),
    EngineSpec("Large", "Large", _loader("Large", "sam2.1_hiera_l.yaml", "./checkpoints/sam2.1_hiera_large.pt"),
               "Slower, slightly more accurate SAM2 model.", 20),
    EngineSpec("Efficient", "Efficient", _loader("Efficient", "efficienttam_s_512x512.yaml", "./checkpoints/efficienttam_s_512x512.pt"),
               "Faster model with lower accuracy.", 30),
)
