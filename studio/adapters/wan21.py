import inspect
import time
import uuid

import torch
from PIL import ImageOps
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

from .base import ModelAdapter
from ..manager import output_dir


class Wan21T2V13BAdapter(ModelAdapter):
    """Wan2.1 T2V 1.3B Diffusers adapter for Colab T4-class GPUs.

    Native target: 832x480, 81 frames, 16 FPS (~5 seconds).
    Export resolutions/aspect ratios are post-processing options.
    """

    def load(self, runtime):
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required for Wan2.1 T2V 1.3B.")

        free_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        if free_gb < 8.5:
            raise RuntimeError(
                f"Only {free_gb:.1f} GB GPU memory is free. "
                "Unload other models and retry with at least about 8.5 GB free."
            )

        try:
            # Keep the Wan VAE in FP32 for more reliable decoding, as advised
            # in the official Diffusers Wan pipeline documentation.
            vae = AutoencoderKLWan.from_pretrained(
                str(self.local_path),
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self.pipeline = WanPipeline.from_pretrained(
                str(self.local_path),
                vae=vae,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            # T4 has 16-bit FP16 support but not native BF16.
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(
                self.pipeline.scheduler.config,
                flow_shift=5.0,
            )
            self.pipeline.enable_sequential_cpu_offload()
            if hasattr(self.pipeline, "vae"):
                if hasattr(self.pipeline.vae, "enable_slicing"):
                    self.pipeline.vae.enable_slicing()
                if hasattr(self.pipeline.vae, "enable_tiling"):
                    self.pipeline.vae.enable_tiling()
        except Exception as exc:
            self.pipeline = None
            raise RuntimeError(
                "Wan2.1 could not be loaded with FP16 + sequential CPU offload on this runtime. "
                "No larger-precision fallback was attempted. "
                f"Details: {type(exc).__name__}: {exc}"
            ) from exc

        self.quantization_label = "FP16 + sequential CPU offload"

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
            "480p": (832, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
        }
        width, height = landscape[resolution]
        if aspect == "9:16":
            return height, width
        return width, height

    def generate(self, prompt, negative_prompt="", resolution="480p",
                 aspect_ratio="16:9", steps=30, guidance=5.0,
                 seed=12345, progress_callback=None):
        if self.pipeline is None:
            raise RuntimeError("Load Wan2.1 T2V 1.3B first.")
        if resolution not in ("360p", "480p", "720p", "1080p"):
            raise ValueError("Choose 360p, 480p, 720p, or 1080p output.")
        if aspect_ratio not in ("16:9", "9:16"):
            raise ValueError("Choose 16:9 or 9:16.")

        # Official Diffusers example uses 81 frames at 16 FPS (~5 seconds).
        frames = 81
        total_steps = int(steps)
        if progress_callback:
            progress_callback.update(2, "Preparing", "Preparing Wan2.1 480p video generation…")

        caps = self.capabilities()
        if not caps.get("callback_on_step_end"):
            raise RuntimeError(
                "Installed Diffusers build does not expose callback_on_step_end. "
                "Update the notebook dependencies and restart the runtime."
            )

        def callback(_pipe, step, _timestep, callback_kwargs):
            completed = min(total_steps, int(step) + 1)
            if progress_callback:
                progress_callback.update(
                    5 + round(88 * completed / total_steps),
                    "Generating",
                    f"Wan2.1 denoising step {completed}/{total_steps}",
                )
            return callback_kwargs

        kwargs = {
            "prompt": str(prompt),
            "negative_prompt": str(negative_prompt) or None,
            "width": 832,
            "height": 480,
            "num_frames": frames,
            "num_inference_steps": total_steps,
            "guidance_scale": float(guidance),
            "generator": torch.Generator(device="cuda").manual_seed(int(seed)),
            "callback_on_step_end": callback,
        }
        if caps.get("callback_on_step_end_tensor_inputs"):
            kwargs["callback_on_step_end_tensor_inputs"] = []

        result = self.pipeline(**kwargs)
        video_frames = getattr(result, "frames", None)
        if not video_frames or not video_frames[0]:
            raise RuntimeError("Wan2.1 returned no video frames.")

        if progress_callback:
            progress_callback.update(95, "Encoding", "Resizing frames and exporting MP4…")

        width, height = self._target_size(resolution, aspect_ratio)
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
        path = output_dir("videos") / f"wan21_13b_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
        export_to_video(converted, str(path), fps=16)
        return {
            "path": str(path),
            "width": width,
            "height": height,
            "fps": 16,
            "duration_seconds": 5,
            "frames": frames,
            "native_resolution": "832x480",
            "quantization": self.quantization_label,
            "note": (
                "480p is the model's native target. 720p and 1080p are upscales; "
                "9:16 is a centre crop from the landscape source."
            ),
        }
