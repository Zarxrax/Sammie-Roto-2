"""Maintainer utility to create the bundled empty-prompt context once.

The resulting file is small and lets every user omit the 12B Gemma checkpoint.
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-encoder", required=True)
    parser.add_argument(
        "--output",
        default="checkpoints/ltx2.5/ltx25_empty_lossless.safetensors",
    )
    args = parser.parse_args()

    from ltx_pipelines.utils.blocks import PromptEncoder
    from ltx_pipelines.utils.model_paths import ModelPaths

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # The split text-encoder checkpoint contains Gemma, the connector weights,
    # config, and tokenizer assets. Avoid requiring a full transformer merely
    # to compute this one cached value.
    model_paths = ModelPaths(
        mode="split",
        transformer_path=None,
        text_encoder_path=args.text_encoder,
        video_vae_path=None,
        audio_vae_path=None,
        duration_head_path=None,
        embeddings_weight_paths=(args.text_encoder,),
    )
    encoder = PromptEncoder(model_paths, torch.bfloat16, device)
    (context,) = encoder([""])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"video_context": context.video_encoding.detach().cpu()}, str(output))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
