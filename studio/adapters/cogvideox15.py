import inspect
import time
import uuid

import torch
from PIL import ImageOps
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
    QuantoConfig as DiffusersQuantoConfig,
)
from diffusers.utils import export_to_video
from transformers import T5EncoderModel, QuantoConfig as TransformersQuantoConfig

from .base import ModelAdapter
from ..manager import output_dir


class CogVideoX15Adapter(ModelAdapter):
    """Experimental CogVideoX1.5-5B T2V path for low-memory Colab GPUs.

    INT8 weight-only quantization is applied separately to the T5 text encoder,
    transformer, and VAE. Activations remain FP16. This is an experimental
    Quanto path; upstream low-memory benchmarks are not a T4 benchmark.
    """

    def load(self, runtime):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for CogVideoX1.5-5B.")
        free_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        if free_gb < 9:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB GPU memory is free. "
                "Unload other GPU work and retry."
            )

        try:
            # Quantize components as they load to avoid keeping the full
            # FP16 pipeline resident in host RAM at the same time.
            text_encoder = T5EncoderModel.from_pretrained(
                str(self.local_path),
                subfolder="text_encoder",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=TransformersQuantoConfig(weights="int8"),
            )
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                str(self.local_path),
                subfolder="transformer",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=DiffusersQuantoConfig(weights_dtype="int8"),
            )
            vae = AutoencoderKLCogVideoX.from_pretrained(
                str(self.local_path),
                subfolder="vae",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                quantization_config=DiffusersQuantoConfig(weights_dtype="int8"),
            )
        except Exception as exc:
            raise RuntimeError(
                "INT8 quantization could not be initialized for one of the "
                "CogVideoX1.5-5B components on this Colab runtime. No FP16 "
                "fallback was attempted because it may exceed available memory. "
                f"Details: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            self.pipeline = CogVideoXPipeline.from_pretrained(
                str(self.local_path),
                text_encoder=text_encoder,
                transformer=transformer,
                vae=vae,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            self.pipeline.enable_sequential_cpu_offload()
            if hasattr(self.pipeline, "vae"):
                if hasattr(self.pipeline.vae, "enable_slicing"):
                    self.pipeline.vae.enable_slicing()
                if hasattr(self.pipeline.vae, "enable_tiling"):
                    self.pipeline.vae.enable_tiling()
        except Exception as exc:
            raise RuntimeError(
                "Could not assemble/offload the quantized CogVideoX1.5 pipeline. "
                f"Details: {type(exc).__name__}: {exc}"
            ) from exc

        self.quantization_label = "INT8 weights: T5 + transformer + VAE; FP16 activations"

    def capabilities(self):
        if self.pipeline is None:
            return {}
        try:
            params = inspect.signature(self.pipeline.__call__).parameters
        except (TypeError, ValueError):
            params = {}
        return {key: key in params for key in (
            "prompt", "negative_prompt", "height", "width", "num_frames",
            "num_inference_steps", "guidance_scale", "generator",
            "callback_on_step_end", "callback_on_step_end_tensor_inputs"
        )}

    @staticmethod
    def _target_size(resolution, aspect):
        landscape = {
            "360p": (640, 360),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
        }
        width, height = landscape[resolution]
        if aspect == "9:16":
            return height, width
        return width, height

    def generate(self, prompt, negative_prompt="", duration_seconds=5, resolution="720p",
                 aspect_ratio="16:9", steps=30, guidance=6.0, seed=12345,
                 progress_callback=None):
        if self.pipeline is None:
            raise RuntimeError("Load CogVideoX1.5-5B first.")
        duration_seconds = int(duration_seconds)
        if duration_seconds not in (5, 10):
            raise ValueError("CogVideoX1.5-5B supports 5 or 10 seconds in this adapter.")
        if resolution not in ("360p", "720p", "1080p"):
            raise ValueError("Choose 360p, 720p, or 1080p output.")
        if aspect_ratio not in ("16:9", "9:16"):
            raise ValueError("Choose 16:9 or 9:16.")

        # Official T2V checkpoint is trained for 1360x768. Export resizing/crop
        # is applied after generation and does not change model-native quality.
        frames = 81 if duration_seconds == 5 else 161
        width, height = self._target_size(resolution, aspect_ratio)
        total_steps = int(steps)

        if progress_callback:
            progress_callback.update(2, "Preparing", "Preparing prompt and latent video frames…")
        if not self.capabilities().get("callback_on_step_end"):
            raise RuntimeError(
                "Installed Diffusers build does not expose callback_on_step_end; update dependencies."
            )
        callback = progress_callback.step_callback(total_steps) if progress_callback else None

        kwargs = {
            "prompt": str(prompt),
            "negative_prompt": str(negative_prompt) or None,
            "width": 1360,
            "height": 768,
            "num_frames": frames,
            "num_inference_steps": total_steps,
            "guidance_scale": float(guidance),
            "generator": torch.Generator(device="cuda").manual_seed(int(seed)),
        }
        if callback is not None:
            kwargs["callback_on_step_end"] = callback
            kwargs["callback_on_step_end_tensor_inputs"] = []

        result = self.pipeline(**kwargs)
        video_frames = getattr(result, "frames", None)
        if not video_frames or not video_frames[0]:
            raise RuntimeError("The model returned no video frames.")

        if progress_callback:
            progress_callback.update(95, "Encoding", "Resizing frames and exporting MP4…")

        target_size = (int(width), int(height))
        converted = [
            ImageOps.fit(
                frame.convert("RGB"),
                target_size,
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            for frame in video_frames[0]
        ]
        path = output_dir("videos") / f"cogvideox15_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
        export_to_video(converted, str(path), fps=16)
        return {
            "path": str(path),
            "width": width,
            "height": height,
            "fps": 16,
            "duration_seconds": duration_seconds,
            "frames": frames,
            "native_resolution": "1360x768",
            "quantization": self.quantization_label,
            "note": (
                "1080p is an upscale from the model's 1360x768 native output. "
                "9:16 is a centre crop from the landscape source."
            ),
        }
