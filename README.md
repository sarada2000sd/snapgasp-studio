# SNAPGASP Studio

A notebook-native, free-tier-first workspace for open-weight image and video generation. The first runtime target is Google Colab with a local-in-notebook control panel and Google Drive storage.

> Colab Free GPU access, GPU type, VRAM, quotas, and session duration are not guaranteed. The app inspects the live runtime and validates model compatibility. No public Gradio tunnel or paid GPU service is used.

## Included

- Notebook-native control panel using ipywidgets.
- Live GPU / VRAM inspection.
- One-model-at-a-time model lifecycle.
- Google Drive cache for weights, generated media, and history.
- Hugging Face model install, update, and uninstall controls.
- Capability inspection against the selected pipeline's actual call signature.
- Diffusers image adapters and CogVideoX 2B text-to-video adapter.
- A catalogue for future adapters, clearly marked as planned until implemented and tested.

## Quick start

1. Open `SNAPGASP_Studio_Colab.ipynb` in Google Colab.
2. Upload the `snapgasp-studio.zip` source archive when prompted.
3. Run the notebook cells in order and mount Google Drive.
4. In Model Manager, install a compatible model, load it, then generate.

The notebook runs controls inside Colab; it does not create a public tunnel or an always-on website. Free GPU capacity is variable.

## Models in this starter

| Model | State |
|---|---|
| Qwen Image | Diffusers image adapter |
| FLUX.1 Schnell | Diffusers image adapter |
| Stable Diffusion 3.5 Medium | Diffusers image adapter; license terms apply |
| CogVideoX 2B | Text-to-video adapter |
| Wan 2.2 TI2V 5B | Planned; adapter not yet implemented |
| HunyuanVideo 1.5 | Planned; adapter not yet implemented |

A catalogue entry is not a promise that a model fits a particular Colab GPU. Large models may fail because of GPU VRAM, system RAM, disk, package compatibility, or upstream restrictions.

## Data and privacy

Weights and outputs are stored in the Google Drive folder set by `SNAPGASP_DATA_DIR`. Prompts and generated files are not uploaded to a SNAPGASP server. Do not commit model weights, tokens, generated private media, or Drive contents to GitHub.

## Licensing

The SNAPGASP Studio application code is MIT-licensed. Model weights, upstream code, datasets, and outputs remain subject to their own licenses and terms. Read each official model card before use, especially commercial use.
