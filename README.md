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

The dashboard uses Colab-native HTML controls and Python callbacks; it does not depend on the third-party ipywidgets renderer. Downloading model weights does not require a GPU; loading and generation do.

## Video controls and progress

- Resolution exports: 360p, 480p, 720p, or 1080p where available.
- Aspect ratio: 16:9 landscape or 9:16 centre crop.
- Duration options are model-dependent.
- Steps, guidance, and seed are shown in plain language.
- The generation panel reports preparation, denoising-step percentage, encoding, Completed, or Failed.

Resolution and aspect ratio controls are export/post-processing settings, not a claim that every model generates at those native dimensions.

## Model catalogue

| Model | State | Notes |
|---|---|---|
| Stable Diffusion 1.5 | Implemented | Image generation test candidate; CreativeML OpenRAIL-M terms apply |
| FLUX.1 Schnell | Implemented | High host RAM needs; not recommended for a 12.7 GB Colab RAM session |
| Qwen Image | Implemented | Estimated hardware needs exceed the current T4/RAM combination |
| Stable Diffusion 3.5 Medium | Implemented | Estimated hardware needs exceed the current T4/RAM combination |
| Wan2.1 T2V 1.3B | Experimental | FP16 + sequential CPU offload; native 832x480, 81 frames, 16 FPS, ~5 seconds; T4 run not yet tested |
| CogVideoX1.5-5B | Experimental | INT8 weights for T5 text encoder, transformer, and VAE via Optimum Quanto + FP16 activations; T4 path experimental |
| CogVideoX-2B | Implemented | FP16 T2V, 720x480, 49 frames, 8 FPS |
| Wan 2.2 TI2V 5B | Planned | Adapter not implemented |
| HunyuanVideo 1.5 | Planned | Adapter not implemented |

### Wan2.1 T2V 1.3B notes

The official Wan documentation describes the 1.3B text-to-video model as a consumer-GPU option, reports an 8.19 GB VRAM requirement, and recommends 480p because 720p results are less stable. The Diffusers example uses 81 frames at 16 FPS (about five seconds). This SNAPGASP adapter uses FP16, an FP32 VAE, and sequential CPU offload; no T4 performance test has been completed. The checkpoint requires approximately 18 GB of free Google Drive storage. 720p and 1080p options are upscales from the 832x480 source; 9:16 is a centre crop, not native portrait generation.

### CogVideoX1.5-5B experimental notes

The upstream model card lists 1360x768 T2V, 81 frames by default, 5 or 10 seconds, and 16 FPS. The optimized low-memory test results are not a guarantee for this Colab T4 setup: the model authors say their tested optimization setup was not tested on non-A100/H100 devices and generally targets Ampere architecture or newer. A T4 is older, so this adapter may fail to load or generate, even when the headline VRAM estimate passes.

This adapter applies INT8 weight-only quantization to the T5 text encoder, transformer, and VAE through Optimum Quanto, with FP16 activations. It is an experimental alternative implementation, not a verified reproduction of the upstream INT8 benchmarks. The Studio shows an architecture warning but allows a best-effort attempt. No FP16 fallback is attempted because it could exceed available memory. The Model Manager checks for approximately 32 GB free Drive space before downloading.

### CogVideoX-2B watermark note

An issue in the upstream CogVideo repository reports a third-party “Pond5” watermark in videos generated with CogVideo 2B. It is a user report, not a formal guarantee that every checkpoint/version will show it; it is one reason SNAPGASP treats this model as a limited test candidate rather than the quality target.

Model memory figures in the catalogue are planning checks, not benchmarks. Actual peak GPU/RAM use may be higher. Free Colab GPU availability, GPU model, RAM, disk, quotas, and session duration are not guaranteed.

## Storage and privacy

- Model weights and generated media are stored under `SNAPGASP_DATA_DIR` on Google Drive.
- No SNAPGASP-hosted server receives prompts or generated files.
- Never commit model weights, generated private media, Hugging Face tokens, or Drive files to GitHub.

## Licensing

This repository's application code is MIT-licensed. Model weights, upstream code, datasets, and outputs remain subject to their own licenses and terms. Wan2.1 T2V 1.3B is Apache 2.0; review the license and comply with its notice requirements. CogVideoX, FLUX, Qwen Image, and Stable Diffusion each have their own terms. See [NOTICES.md](NOTICES.md).
