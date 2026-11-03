import inspect, time, uuid, torch
from PIL import ImageOps
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from .base import ModelAdapter
from ..manager import output_dir

class CogVideoXAdapter(ModelAdapter):
    """CogVideoX-2B T2V; native 720x480, 49 frames, 8 FPS."""
    def load(self,runtime):
        self.pipeline=CogVideoXPipeline.from_pretrained(str(self.local_path),torch_dtype=torch.float16)
        if hasattr(self.pipeline,"vae"):
            if hasattr(self.pipeline.vae,"enable_slicing"): self.pipeline.vae.enable_slicing()
            if hasattr(self.pipeline.vae,"enable_tiling"): self.pipeline.vae.enable_tiling()
        if torch.cuda.is_available() and hasattr(self.pipeline,"enable_sequential_cpu_offload"):
            self.pipeline.enable_sequential_cpu_offload()
        elif torch.cuda.is_available() and hasattr(self.pipeline,"enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload()
        else:
            raise RuntimeError("CogVideoX generation requires a CUDA GPU.")
    def capabilities(self):
        if self.pipeline is None: return {}
        try: params=inspect.signature(self.pipeline.__call__).parameters
        except (TypeError,ValueError): params={}
        return {k:k in params for k in ("prompt","negative_prompt","height","width","num_frames","num_inference_steps","guidance_scale","generator","callback_on_step_end","callback_on_step_end_tensor_inputs")}
    @staticmethod
    def target_size(resolution, aspect):
        sizes={"360p":(640,360),"720p":(1280,720),"1080p":(1920,1080)}
        w,h=sizes[resolution]
        return (h,w) if aspect=="9:16" else (w,h)
    def generate(self,prompt,negative_prompt="",width=720,height=480,frames=49,fps=8,steps=30,guidance=6,seed=0,
                 resolution="720p",aspect_ratio="16:9",progress_callback=None):
        if self.pipeline is None: raise RuntimeError("Load CogVideoX first.")
        if (int(width),int(height),int(frames),int(fps)) != (720,480,49,8):
            raise ValueError("CogVideoX-2B source generation is fixed at 720x480, 49 frames, 8 FPS.")
        if progress_callback:
            progress_callback.update(2,"Preparing","Preparing 6-second CogVideoX-2B generation…")
        args={"prompt":prompt,"negative_prompt":negative_prompt or None,"width":720,"height":480,
              "num_frames":49,"num_inference_steps":int(steps),"guidance_scale":float(guidance),
              "generator":torch.Generator(device="cuda").manual_seed(int(seed))}
        if progress_callback and self.capabilities().get("callback_on_step_end"):
            args["callback_on_step_end"]=progress_callback.step_callback(int(steps))
            args["callback_on_step_end_tensor_inputs"]=[]
        result=self.pipeline(**args)
        frames_out=getattr(result,"frames",None)
        if not frames_out or not frames_out[0]: raise RuntimeError("CogVideoX returned no frames.")
        if progress_callback: progress_callback.update(95,"Encoding","Resizing frames and exporting MP4…")
        target=self.target_size(resolution,aspect_ratio)
        converted=[ImageOps.fit(frame.convert("RGB"),target,method=Image.Resampling.LANCZOS,centering=(0.5,0.5)) for frame in frames_out[0]]
        path=output_dir("videos")/f"cogvideox2b_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"
        export_to_video(converted,str(path),fps=8)
        return {"path":str(path),"width":target[0],"height":target[1],"fps":8,"duration_seconds":6,
                "frames":49,"native_resolution":"720x480","quantization":"FP16","note":"Output size is resized/cropped from the model's native 720x480 frames."}
