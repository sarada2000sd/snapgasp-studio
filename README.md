# SNAPGASP Studio

A notebook-native AI image/video generation workspace targeting Google Colab Free and Google Drive. It does not start a public Gradio tunnel or depend on a paid GPU server.

## Quick start

1. Open [SNAPGASP_Studio_Colab.ipynb](SNAPGASP_Studio_Colab.ipynb) in Google Colab.
2. Select a GPU runtime when one is available.
3. Run notebook cells from top to bottom. The first cell clones or updates this repository automatically.
4. Mount Google Drive when prompted.
5. Open Model Manager. Choose a model marked as implemented and compatible with the detected GPU, then install and load it.
6. Generate media. Outputs and model cache are saved to your Drive folder.

## Current T4 guidance

The first video model to try on a Colab T4 is CogVideoX-2B, using the Diffusers low-memory path. The upstream Diffusers guide reports approximately 12.5 GB VRAM for its FP16 single-GPU inference setup with optimizations. A T4 runtime reporting about 14.6 GB has limited headroom; generation may still fail due to memory, package versions, runtime availability, or session limits. The app checks estimated GPU and system RAM before allowing an install/load attempt.

CogVideoX-2B is configured here for 720x480, 49 frames, 8 FPS. Do not change these until the adapter exposes additional verified modes.

## Model catalogue

| Model | Adapter | Notes |
|---|---|---|
| Qwen Image | Diffusers | High memory; not a fit for the current T4/RAM combination |
| FLUX.1 Schnell | Diffusers | Large host-memory requirement; not recommended for current Colab RAM |
| Stable Diffusion 3.5 Medium | Diffusers | Estimated requirement exceeds current T4 |
| CogVideoX 2B | Diffusers | First video model to test on T4 with CPU offload and VAE slicing/tiling |
| Wan 2.2 TI2V 5B | Planned | Adapter not implemented |
| HunyuanVideo 1.5 | Planned | Adapter not implemented |

The catalogue is not a guarantee of compatibility. Model memory estimates are conservative guidance, not benchmark results. Free Colab GPU availability, GPU model, RAM, quotas, and session duration are not guaranteed.

## Storage and privacy

- Model weights and generated media are stored under `SNAPGASP_DATA_DIR` on Google Drive.
- No SNAPGASP-hosted server receives prompts or generated files.
- Never commit model weights, generated private media, Hugging Face tokens, or Drive files to GitHub.

## Licensing

This repository's application code is MIT-licensed. Model weights, upstream code, datasets, and outputs remain subject to their own licenses and terms. Review each official model card and license before use, especially for commercial/stock distribution.

## Local checks

Run `python scripts/selfcheck.py` to check the model catalogue without downloading model weights.
