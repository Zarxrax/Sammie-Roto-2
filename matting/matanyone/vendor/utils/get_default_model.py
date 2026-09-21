"""
Helper functions for loading MatAnyone models from local checkpoints.
"""

import torch

from matting.matanyone.vendor.config import MatAnyoneConfig
from matting.matanyone.vendor.model.matanyone import MatAnyone


def get_matanyone_model(ckpt_path, device=None) -> MatAnyone:
    cfg = MatAnyoneConfig()

    if device is not None:
        matanyone = MatAnyone(cfg, single_object=True).to(device).eval()
        model_weights = torch.load(
            ckpt_path,
            map_location=device,
            weights_only=True,
        )
    else:
        matanyone = MatAnyone(cfg, single_object=True).cuda().eval()
        model_weights = torch.load(
            ckpt_path,
            weights_only=True,
        )

    matanyone.load_weights(model_weights)

    return matanyone