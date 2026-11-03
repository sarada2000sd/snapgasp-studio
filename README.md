# SNAPGASP Studio

A notebook-native AI image/video workspace targeting Google Colab Free and Google Drive. It does not start a public Gradio tunnel or depend on a paid GPU server.

## Quick start

1. Open [SNAPGASP_Studio_Colab.ipynb](SNAPGASP_Studio_Colab.ipynb) in Google Colab.
2. Select a GPU runtime when one is available.
3. Run notebook cells from top to bottom. The first cell clones or updates this repository automatically.
4. Mount Google Drive when prompted.
5. Open Model Manager. Choose a model marked as implemented and compatible with the detected GPU, then install and load it.
6. Generate media. Outputs and model cache are saved to your Drive folder.

The Studio dashboard uses Colab-native HTML controls and Python callbacks; it does not depend on the third-party ipywidgets renderer.

## Current T4 guidance

- Image test candidate: Stable Diffusion 1.5 at 512x512. Estimated minimum 6 GB VRAM and 8 GB system RAM for this adapter's offload path. This is a planning estimate, not a benchmark.
- Video test candidate: CogVideoX-2B using FP16 with CPU offload and VAE slicing/tiling. The upstream Diffusers inference guide reports about 12.5 GB VRAM in its optimized setup. A T4 reporting about 14.6 GB has limited headroom, so generation can still fail.
- CogVideoX-2B is configured here for 720x480, 49 frames, 8 FPS.

## Model catalogue

| Model | Adapter | Notes |
|---|---|---|
| Stable Diffusion 1.5 | Diffusers | First image model to test on T4; CreativeML OpenRAIL-M terms apply |
| FLUX.1 Schnell | Diffusers | Large host-memory requirement; not recommended for current Colab RAM |
| Qwen Image | Diffusers | High memory; not a fit for the current T4/RAM combination |
| Stable Diffusion 3.5 Medium | Diffusers | Estimated requirement exceeds current T4 |
| CogVideoX 2B | Diffusers | First video model to test on T4 with low-memory options |
| Wan 2.2 TI2V 5B | Planned | Adapter not implemented |
| HunyuanVideo 1.5 | Planned | Adapter not implemented |

The catalogue is not a guarantee of compatibility. Memory estimates are guidance only. Free Colab GPU availability, GPU model, RAM, quotas, and session duration are not guaranteed.

## Storage and privacy

- Model weights and generated media are stored under `SNAPGASP_DATA_DIR` on Google Drive.
- No SNAPGASP-hosted server receives prompts or generated files.
- Never commit model weights, generated private media, Hugging Face tokens, or Drive files to GitHub.

## Licensing

This repository's application code is MIT-licensed. Model weights, upstream code, datasets, and outputs remain subject to their own licenses and terms. The Stable Diffusion 1.5 model card identifies the CreativeML OpenRAIL-M license; review its full restrictions before commercial or stock use. See [model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and [notices](NOTICES.md).
