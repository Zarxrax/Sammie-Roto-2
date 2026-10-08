# LTX 2.5 Alpha Gen

This engine creates a full-frame alpha matte directly from RGB frames. It does
not use Sammie's segmentation masks or text prompting.

The runtime downloads only the Q4_K_M distilled transformer, video VAE, and Alpha Gen
IC-LoRA. It deliberately omits Gemma, the audio VAE, duration head, and spatial
upscaler. `ltx25_empty_lossless.safetensors` is the small ComfyUI lossless
cache of the projected empty-prompt conditioning placed in `checkpoints/ltx2.5`.
The connector bundled in the GGUF turns it into the final video context at
startup, so Gemma is not needed at runtime.

The transformer uses a user-selected quant from the public
`realrebelai/LTX-2.5_GGUFs` repository; the video VAE uses the public
`comfyicu/LTX-2.5` mirror. Neither requires authentication. On first use,
Sammie downloads the VAE and presents Q2 through Q8 transformer choices with
download size, estimated base memory, and quality tradeoffs. The Alpha Gen
LoRA remains in Lightricks' gated repository and is handled last. Sammie then
presents the manual-install and authenticated-download options for the LoRA.
Automatic download requires accepting the repository terms and a read-only
Hugging Face token; the token is validated and stored only in Hugging Face's
standard user login cache.

The vendored LTX runtime is pinned to upstream commit
`9ec55f9f22798a3198d9c923856824821bc3317e` (release 1.4.2). Its package-level
Torch indexes were removed so Sammie's platform-specific Torch selection stays
authoritative. SciPy is optional because this engine uses the fixed distilled
sigma schedule rather than `BetaScheduler`.

Sammie's local GGUF backend keeps the base weights packed and dequantizes one
linear layer at a time. The Alpha Gen adapter is evaluated as a separate low
rank term, so applying it does not expand the complete transformer to BF16.
The dequantization kernels are adapted from ComfyUI-GGUF under Apache-2.0.

On macOS, reference-video encoding uses Diffusers' LTX 2.5 encoder with the
same VAE checkpoint. Its tensor mapping is numerically identical to the LTX
Core encoder, while avoiding non-finite output from the Core encoder on MPS.
