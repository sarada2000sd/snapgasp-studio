# SNAPGASP Studio

A notebook-native AI image/video workspace targeting Google Colab Free and Google Drive. It does not start a public Gradio tunnel or depend on a paid GPU server.

## Quick start

1. Open [SNAPGASP_Studio_Colab.ipynb](SNAPGASP_Studio_Colab.ipynb) in Google Colab.
2. Select a GPU runtime when one is available.
3. Run notebook cells from top to bottom. The first cell clones or updates this repository automatically.
4. Mount Google Drive when prompted.
5. Open Model Manager. Check free Drive storage before downloading large checkpoints.
6. Select a model marked Implemented or Experimental, install it, load it, then generate.
7. Generated media and model caches are saved under `SNAPGASP_DATA_DIR` in Drive.

The dashboard uses Colab-native HTML controls and Python callbacks; it does not depend on the third-party ipywidgets renderer.

## Video controls and progress

- Resolution: 360p, 720p, or 1080p export.
- Aspect ratio: 16:9 landscape or 9:16 centre crop.
- Duration: dynamic per model.
- Steps, guidance, and seed are shown in plain language.
- The generation panel reports preparation, real denoising-step percentage, encoding, Completed, or Failed.

Resolution and aspect ratio controls are export/post-processing settings, not a claim that every model generates at those native dimensions. For CogVideoX1.5-5B, the native T2V source is 1360x768. 1080p is an upscale and 9:16 is a centre crop; neither creates extra model detail or native portrait composition. CogVideoX-2B's source is 720x480, 49 frames, 8 FPS, about 6 seconds; the same export presets are applied after generation.

## Model catalogue

| Model | State | Notes |
|---|---|---|
| Stable Diffusion 1.5 | Implemented | First image test candidate on a 14.6 GB T4; CreativeML OpenRAIL-M terms apply |
| FLUX.1 Schnell | Implemented | High host RAM needs; not recommended for a 12.7 GB Colab RAM session |
| Qwen Image | Implemented | Estimated hardware needs exceed the current T4/RAM combination |
| Stable Diffusion 3.5 Medium | Implemented | Estimated hardware needs exceed the current T4/RAM combination |
| CogVideoX1.5-5B | Experimental | INT8 transformer weights via Optimum Quanto + FP16 activations; T4 path is unbenchmarked |
| CogVideoX-2B | Implemented | FP16 T2V, 720x480, 49 frames, 8 FPS |
| Wan 2.2 TI2V 5B | Planned | Adapter not implemented |
| HunyuanVideo 1.5 | Planned | Adapter not implemented |

### CogVideoX1.5-5B experimental notes

The upstream model card lists 1360x768 T2V, 81 frames by default, 5 or 10 seconds, and 16 FPS. It documents quantization options that can lower memory needs. The optimized low-memory test results are not a guarantee for this Colab T4 setup: the model authors say their tested optimization setup was not tested on non-A100/H100 devices and generally targets NVIDIA Ampere architecture or newer. A T4 is older, so this adapter may fail to load or generate, even when the headline VRAM estimate passes.

This adapter quantizes the transformer weights to INT8 with Optimum Quanto, while leaving activations and other pipeline components in FP16. It is not the complete all-components INT8 recipe shown by the upstream project. No FP16 fallback is attempted because it could exceed the available memory. The Model Manager checks for approximately 32 GB free Drive space before downloading; this is a safety estimate, not an exact upstream file-size guarantee.

Model memory figures in the catalogue are planning checks, not benchmarks. Actual peak GPU/RAM use may be higher. Free Colab GPU availability, GPU model, RAM, disk, quotas, and session duration are not guaranteed.

## Storage and privacy

- Model weights and generated media are stored under `SNAPGASP_DATA_DIR` on Google Drive.
- No SNAPGASP-hosted server receives prompts or generated files.
- Never commit model weights, generated private media, Hugging Face tokens, or Drive files to GitHub.

## Licensing

This repository's application code is MIT-licensed. Model weights, upstream code, datasets, and outputs remain subject to their own licenses and terms. Review each official model card and license before commercial or stock use. CogVideoX is released under its upstream CogVideoX license; Stable Diffusion 1.5 identifies CreativeML OpenRAIL-M. See [NOTICES.md](NOTICES.md).
